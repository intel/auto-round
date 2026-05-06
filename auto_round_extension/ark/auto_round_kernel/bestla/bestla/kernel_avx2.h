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

namespace bestla {
namespace kernel {
namespace avx2 {
#if CompileAVX2()
#if defined(__INTEL_LLVM_COMPILER)
#pragma clang attribute push(__attribute__((target("avx2,fma,f16c"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "fma", "f16c")
#endif

static inline void zero_reg() { _mm256_zeroupper(); }

static inline __m256i unpack_4bits(void* srcptr, __m256i mask) {
  auto raw_data = _mm_loadu_si128(reinterpret_cast<__m128i*>(srcptr));
  auto ymm0 = _mm256_cvtepu8_epi16(raw_data);
  auto ymm1 = _mm256_slli_epi16(ymm0, 4);
  ymm0 = _mm256_or_si256(ymm0, ymm1);
  ymm0 = _mm256_and_si256(ymm0, mask);
  return ymm0;
}

static inline __m256i unpack_2bits(utils::bit2x4* ptr, const __m256i& vshift_y, const __m256i& vmask0_y,
                                   const __m256i& vsfhl_mask_y, const __m256i& vorder_y) {
  auto vraw_x = _mm_loadl_epi64((const __m128i*)ptr);
  auto vsrc_y = _mm256_broadcastq_epi64(vraw_x);
  auto vordered_y = _mm256_permutevar8x32_epi32(vsrc_y, vorder_y);
  auto vs_y = _mm256_srlv_epi32(vordered_y, vshift_y);
  auto v2_y = _mm256_and_si256(vs_y, vmask0_y);
  auto vout_y = _mm256_shuffle_epi8(v2_y, vsfhl_mask_y);
  return vout_y;
}

static inline __m256i unpack_1bits(utils::bit1x8* ptr, const __m256i& bit1Shift_1, const __m256i& bit1Mask,
                                   const __m256i& bit1Shift_2, const __m256i& highMask) {
  auto bit1x32 = _mm256_set1_epi32(*(int*)ptr);
  bit1x32 = _mm256_srlv_epi32(bit1x32, bit1Shift_1);
  bit1x32 = _mm256_and_si256(bit1x32, bit1Mask);
  bit1x32 = _mm256_mullo_epi32(bit1x32, bit1Shift_2);
  bit1x32 = _mm256_and_si256(highMask, bit1x32);
  return bit1x32;
}

inline __m256 ymm_cvt_bf16_fp32(__m128i vbf16) {
  auto vf32 = _mm256_cvtepu16_epi32(vbf16);
  return _mm256_castsi256_ps(_mm256_slli_epi32(vf32, 16));
}

inline __m256 ymm_cvt_fp16_fp32(__m128i vfp16) { return _mm256_cvtph_ps(vfp16); }

inline __m128i ymm_cvt_fp32_fp16(__m256 vfp32) { return _mm256_cvtps_ph(vfp32, _MM_FROUND_TO_NEAREST_INT); }

inline __m128i ymm_cvtepi32_epi16(__m256i src) {
  const auto shuffle_mask_32_to_16 = _mm256_set_epi8(13, 12, 9, 8, 5, 4, 1, 0, 13, 12, 9, 8, 5, 4, 1, 0, 13, 12, 9, 8,
                                                     5, 4, 1, 0, 13, 12, 9, 8, 5, 4, 1, 0);
  __m256i trunc_elements = _mm256_shuffle_epi8(src, shuffle_mask_32_to_16);
  __m256i ordered = _mm256_permute4x64_epi64(trunc_elements, 0x58);
  __m128i result = _mm256_castsi256_si128(ordered);
  return result;
}

static const uint8_t avx2_bf16_convert_maigc_num[32] = {
    0x02, 0x03, 0x06, 0x07, 0x0a, 0x0b, 0x0e, 0x0f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x02, 0x03, 0x06, 0x07, 0x0a, 0x0b, 0x0e, 0x0f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80};

static inline __m128i cvt_fp32_to_bf16(const __m256 src) {
  const auto bf16_and_helper = _mm256_set1_epi32(0X00000001);
  const auto bf16_add_helper = _mm256_set1_epi32(0x00007FFF);
  auto shuffle_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(avx2_bf16_convert_maigc_num));
  auto round_bias = _mm256_castps_si256(src);
  round_bias = _mm256_and_si256(bf16_and_helper, _mm256_srli_si256(round_bias, 2));
  round_bias = _mm256_add_epi32(round_bias, bf16_add_helper);
  auto round_fp32_v = _mm256_add_epi32(_mm256_castps_si256(src), round_bias);
  __m256i trunc_elements = _mm256_shuffle_epi8(round_fp32_v, shuffle_v);
  __m256i ordered = _mm256_permute4x64_epi64(trunc_elements, 0x58);
  return _mm256_castsi256_si128(ordered);
}

inline __m128i ymm_cvt_fp32_bf16(const __m256& vfp32) {
#if FP32_BF16_FAST
  return ymm_cvtepi32_epi16(_mm256_bsrli_epi128(_mm256_castps_si256(vfp32), 2));
#else
  return cvt_fp32_to_bf16(vfp32);
#endif
}

static inline __m256i load_s8_s32(int8_t* srcptr) {
  auto xmm = _mm_loadl_epi64(reinterpret_cast<__m128i*>(srcptr));
  auto ymm = _mm256_cvtepi8_epi32(xmm);
  return ymm;
}

static inline __m256 load_bf16_fp32(const utils::bf16* srcptr) {
  auto tmp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcptr));
  auto vf32 = ymm_cvt_bf16_fp32(tmp);
  return vf32;
}

static inline __m256 load_fp16_fp32(const utils::fp16* srcptr) {
  auto tmp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcptr));
  auto vf32 = ymm_cvt_fp16_fp32(tmp);
  return vf32;
}

template <typename T>
static inline __m256 load_T_fp32(const T* srcptr) {
  __m256 vtmp;
  if constexpr (std::is_same_v<T, float>) {
    vtmp = _mm256_loadu_ps(srcptr);
  } else if constexpr (std::is_same_v<T, utils::bf16>) {
    vtmp = load_bf16_fp32(srcptr);
  } else if constexpr (std::is_same_v<T, utils::fp16>) {
    vtmp = load_fp16_fp32(srcptr);
  } else {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, utils::bf16> || std::is_same_v<T, utils::fp16>);
  }
  return vtmp;
}

static inline __m256 load_s8_fp32(int8_t* srcptr) {
  auto src_y = load_s8_s32(srcptr);
  auto dst_y = _mm256_cvtepi32_ps(src_y);
  return dst_y;
}

template <typename T>
static inline void store_fp32_T(const __m256& src_y, T* dstptr) {
  if constexpr (std::is_same_v<T, utils::bf16>) {
    auto xmm = ymm_cvt_fp32_bf16(src_y);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dstptr), xmm);
  } else if constexpr (std::is_same_v<T, float>) {
    _mm256_storeu_ps(dstptr, src_y);
  } else if constexpr (std::is_same_v<T, utils::fp16>) {
    auto xmm = ymm_cvt_fp32_fp16(src_y);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dstptr), xmm);
  } else {
    assert(0);
  }
}

template <typename T>
static inline void convert_s8_fp_v8(T* dstptr, int8_t* srcptr) {
  auto src_fp_y = load_s8_fp32(srcptr);
  store_fp32_T(src_fp_y, dstptr);
}

template <bool IsAsym = false>
static inline __m256 dequant_s8_fp(int8_t* srcptr, __m256 vscales, __m256i vzps = __m256i()) {
  auto src_s32_y = load_s8_s32(srcptr);
  if constexpr (IsAsym) src_s32_y = _mm256_sub_epi32(src_s32_y, vzps);
  auto src_fp_y = _mm256_cvtepi32_ps(src_s32_y);
  src_fp_y = _mm256_mul_ps(src_fp_y, vscales);
  return src_fp_y;
}

template <int N, bool IsAsym, typename DstT>
static inline void dequant_s8_N_avx2(DstT* dstptr, int8_t* srcptr, __m256* vscales, __m256i* vzps = nullptr) {
  static_assert(N % 8 == 0);
  int constexpr VLoop = N / 8;
  for (int iv = 0; iv < VLoop; iv += 1) {
    __m256 dq_f32_y;
    if constexpr (IsAsym) {
      dq_f32_y = dequant_s8_fp<IsAsym>(srcptr, vscales[iv], vzps[iv]);
    } else {
      dq_f32_y = dequant_s8_fp<IsAsym>(srcptr, vscales[iv]);
    }
    store_fp32_T(dq_f32_y, dstptr + iv * 8);
  }
}

static inline __m256i load_zp_epi8_broadcast_epi16_v16(int8_t* zpptr, const __m256i& vindex) {
  auto v_zp_x = _mm_loadu_si128((const __m128i*)zpptr);
  auto v_zp_y = _mm256_cvtepi8_epi16(v_zp_x);
  auto v_zp_y_cast = _mm256_shuffle_epi8(v_zp_y, vindex);
  return v_zp_y_cast;
}

static inline __m256i load_zp_epi8_broadcast_epi16(int8_t* zpptr, const __m256i& vindex) {
  auto v_zp_x = _mm_loadu_si128((const __m128i*)zpptr);
  auto v_zp_y = _mm256_cvtepi8_epi16(v_zp_x);
  auto v_zp_y_cast = _mm256_shuffle_epi8(v_zp_y, vindex);
  return v_zp_y_cast;
}

static inline __m256i load_zp_epi8_broadcast_epi32(int8_t* zpptr, const __m256i& vindex) {
  auto v_zp_x = _mm_loadl_epi64((const __m128i*)zpptr);
  auto v_zp_y = _mm256_cvtepi8_epi32(v_zp_x);
  auto v_zp_y_cast = _mm256_shuffle_epi8(v_zp_y, vindex);
  return v_zp_y_cast;
}

// vout= {vsrc.f32[0],vsrc.f32[0],...,vsrc.f32[4],vsrc.f32[4]}
template <bool LowBits>
static inline __m256 broadcast_ps_1_2(__m256 vsrc_y, const __m256i& vshuf_index_y) {
  __m256 tmp;
  if constexpr (LowBits) {
    tmp = _mm256_permute2f128_ps(vsrc_y, vsrc_y, 0);
  } else {
    tmp = _mm256_permute2f128_ps(vsrc_y, vsrc_y, 17);
  }
  auto tmpi = _mm256_castps_si256(tmp);

  auto out = _mm256_shuffle_epi8(tmpi, vshuf_index_y);
  return _mm256_castsi256_ps(out);
}

template <bool LowBits>
static inline __m256i broadcast_epi32_1_2(__m256i vsrc_y, const __m256i& vshuf_index_y) {
  return _mm256_castps_si256(broadcast_ps_1_2<LowBits>(_mm256_castsi256_ps(vsrc_y), vshuf_index_y));
}

inline BTLA_CODE dq8_get_fp_scale(uint8_t* src, float* dst, int row, int col, int scale_offset, int dq_blk,
                                  int dq_offset_idx, float* dq_scale, int src_stride, int dst_stride, bool zeropadding,
                                  int mN) {
  auto head_proc_num = utils::updiv(scale_offset, 8) * 8 - scale_offset;
  auto ymm_dq_offset = _mm256_set1_ps(dq_scale[dq_offset_idx]);

  auto get_fp_scale_ref = [&](int proc_src_num, int scale_offset, uint8_t* src, float* dst) {
    auto dq_s_idx = scale_offset / dq_blk;
    for (int j = 0; j < col; j++) dst[j] = dq8_bnb_LUT[src[j]] * dq_scale[dq_s_idx] + dq_scale[dq_offset_idx];
  };

  auto get_fp_scale_avx2 = [&](int scale_offset, uint8_t* src, float* dst) {
    auto dq_s_idx = scale_offset / dq_blk;
    auto ymm_dq_scale = _mm256_set1_ps(dq_scale[dq_s_idx]);
    float tmp[8];
    for (int i = 0; i < 8; i++) tmp[i] = dq8_bnb_LUT[src[i]];
    __m256 fp32_dq_ymm = _mm256_loadu_ps(tmp);
    auto fymm = _mm256_mul_ps(fp32_dq_ymm, ymm_dq_scale);
    fymm = _mm256_add_ps(fymm, ymm_dq_offset);
    _mm256_storeu_ps(dst, fymm);
  };

  for (int i = 0; i < row; i++) {
    if (head_proc_num > col) {
      get_fp_scale_ref(col, scale_offset + i * mN, src + i * src_stride, dst + i * dst_stride);
    } else {
      get_fp_scale_ref(head_proc_num, scale_offset + i * mN, src + i * src_stride, dst + i * dst_stride);
      auto scale_offset_iter = scale_offset + i * mN + head_proc_num;
      uint8_t* src_iter_ptr = src + head_proc_num;
      float* dst_iter_ptr = dst + head_proc_num;
      auto body_loop = (col - head_proc_num) / 8;
      auto tail_proc_num = (col - head_proc_num) % 8;
      int ii = 0;
      for (; ii < body_loop; ii++) {
        get_fp_scale_avx2(scale_offset_iter + ii * 8, src_iter_ptr + i * src_stride + ii * 8,
                          dst_iter_ptr + i * dst_stride + ii * 8);
      }
      if (tail_proc_num > 0) {
        get_fp_scale_ref(tail_proc_num, scale_offset_iter + ii * 8, src_iter_ptr + i * src_stride + ii * 8,
                         dst_iter_ptr + i * dst_stride + ii * 8);
      }
    }
  }
  if (zeropadding) assert(0);
  return BTLA_CODE::Success;
}

static inline BTLA_CODE alphabeta_f32_f32(const float alpha, const float* srcptr, const int srcstep, const float beta,
                                          const float* src1ptr, const int src1step, float* dstptr, const int dststep,
                                          const int M, const int N) {
  int constexpr Vlen = 8;
  auto vN = utils::padto_le(N, Vlen);
  auto valpha = _mm256_set1_ps(alpha);
  auto vbeta = _mm256_set1_ps(beta);

  for (int i = 0; i < M; i++) {
    int j = 0;
    if (beta != 0.f) {
      for (; j < vN; j += Vlen) {
        auto vsrc = _mm256_loadu_ps(srcptr + i * srcstep + j);
        auto vsrc1 = _mm256_loadu_ps(src1ptr + i * src1step + j);
        auto vdst = _mm256_mul_ps(valpha, vsrc);
        vdst = _mm256_fmadd_ps(vbeta, vsrc1, vdst);
        _mm256_storeu_ps(dstptr + i * dststep + j, vdst);
      }
      for (; j < N; j += 1) {
        dstptr[i * dststep + j] = alpha * srcptr[i * srcstep + j] + beta * src1ptr[i * src1step + j];
      }
    } else {
      for (; j < vN; j += Vlen) {
        auto vsrc = _mm256_loadu_ps(srcptr + i * srcstep + j);
        auto vdst = _mm256_mul_ps(valpha, vsrc);
        _mm256_storeu_ps(dstptr + i * dststep + j, vdst);
      }
      for (; j < N; j += 1) {
        dstptr[i * dststep + j] = alpha * srcptr[i * srcstep + j];
      }
    }
  }
  return BTLA_CODE::Success;
}

template <typename SCAB_T, int NTILE>
static inline BTLA_CODE dequant_s32_fp32_N(const int32_t* srcptr, const int srcstep, float* dstptr, const int dststep,
                                           const int row, const float* scaleA, const int ldsa, const SCAB_T* scaleB) {
  int constexpr VLen = 8;
  static_assert(NTILE % VLen == 0);
  int constexpr NRegs = NTILE / VLen;
  __m256 vwscale[NRegs];
  for (int ic = 0; ic < NRegs; ic++) {
    vwscale[ic] = load_T_fp32(scaleB + ic * VLen);
  }
  for (int irow = 0; irow < row; irow++) {
    auto scale = scaleA[irow * ldsa];
    auto valpha = _mm256_set1_ps(scale);
    for (int ic = 0; ic < NRegs; ic++) {
      auto vscale = _mm256_mul_ps(valpha, vwscale[ic]);
      auto vsrcd = _mm256_loadu_si256((const __m256i*)(srcptr + irow * srcstep + ic * VLen));
      auto vsrc = _mm256_cvtepi32_ps(vsrcd);
      vsrc = _mm256_mul_ps(vsrc, vscale);
      _mm256_storeu_ps(dstptr + irow * dststep + ic * VLen, vsrc);
    }
  }
  return BTLA_CODE::Success;
}

template <typename SCAB_T>
static inline BTLA_CODE dequant_s32_fp32(const int32_t* srcptr, const int srcstep, float* dstptr, const int dststep,
                                         const int row, const int col, const float* scaleA, const int ldsa,
                                         const SCAB_T* scaleB) {
  int col24 = utils::padto_le(col, 24);
  int col16 = utils::padto_le(col, 16);
  int col8 = utils::padto_le(col, 8);
  int icol = 0;
  for (; icol < col24; icol += 24) {
    dequant_s32_fp32_N<SCAB_T, 24>(srcptr + icol, srcstep, dstptr + icol, dststep, row, scaleA, ldsa, scaleB + icol);
  }
  if (icol + 16 <= col16) {
    for (; icol < col16; icol += 16) {
      dequant_s32_fp32_N<SCAB_T, 16>(srcptr + icol, srcstep, dstptr + icol, dststep, row, scaleA, ldsa, scaleB + icol);
    }
  }
  if (icol + 8 <= col8) {
    for (; icol < col8; icol += 8) {
      dequant_s32_fp32_N<SCAB_T, 8>(srcptr + icol, srcstep, dstptr + icol, dststep, row, scaleA, ldsa, scaleB + icol);
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

static inline BTLA_CODE remove_act_zeropoint_bias(float* accptr, int ldacc, int row, int col, uint8_t* zps,
                                                  float* scales, int lds, const float* reduce) {
  int constexpr VLen = 8;
  auto col8 = utils::padto_le(col, VLen);
  for (int i = 0; i < row; i++) {
    auto zpf = static_cast<float>(zps[i * lds]) * scales[i * lds];
    int j = 0;
    auto vzp = _mm256_set1_ps(-zpf);
    for (; j < col8; j += VLen) {
      auto vreduce = _mm256_loadu_ps(reduce + j);
      auto vacc = _mm256_loadu_ps(&accptr[i * ldacc + j]);
      vacc = _mm256_fmadd_ps(vzp, vreduce, vacc);
      _mm256_storeu_ps(&accptr[i * ldacc + j], vacc);
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
  int constexpr VLen = 8;
  auto col8 = utils::padto_le(col, VLen);
  const int32_t mask[] = {-1, -1, 0, 0};
  for (int i = 0; i < row; i++) {
    auto vreduce = _mm256_set1_ps(-reduce[i * lds]);
    int j = 0;
    for (; j < col8; j += VLen) {
      auto vzp_s32 = _mm256_cvtepi8_epi32(_mm_maskload_epi32(reinterpret_cast<const int*>(zps + j),
                                                             _mm_loadu_si128(reinterpret_cast<const __m128i*>(mask))));
      auto vzp_f32 = _mm256_cvtepi32_ps(vzp_s32);
      auto vzp = _mm256_mul_ps(vzp_f32, _mm256_loadu_ps(scales + j));
      auto vacc = _mm256_loadu_ps(&accptr[i * ldacc + j]);
      vacc = _mm256_fmadd_ps(vzp, vreduce, vacc);
      _mm256_storeu_ps(&accptr[i * ldacc + j], vacc);
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
  int constexpr VLen = 8;
  auto col16 = utils::padto_le(col, VLen);
  auto vk = _mm256_set1_epi32(k);
  for (int i = 0; i < row; i++) {
    auto vreducea = ZPB ? _mm256_set1_epi32(reducea[i * lda]) : _mm256_set1_epi32(0);
    auto vzpa = ZPA ? _mm256_set1_epi32(int(zpa[i * lda])) : _mm256_set1_epi32(0);
    int j = 0;
    for (; j < col16; j += VLen) {
      auto vzpb = ZPB ? load_s8_s32(zpb + j) : _mm256_set1_epi32(0);
      auto vreduceb = ZPA ? _mm256_loadu_si256((__m256i*)(reduceb + j)) : _mm256_set1_epi32(0);
      auto vacc = _mm256_loadu_si256((__m256i*)&accptr[i * ldacc + j]);
      __m256i tmp;
      if constexpr (ZPA) {
        tmp = _mm256_mullo_epi32(vzpa, vreduceb);
        vacc = _mm256_sub_epi32(vacc, tmp);
      }
      if constexpr (ZPB) {
        tmp = _mm256_mullo_epi32(vzpb, vreducea);
        vacc = _mm256_sub_epi32(vacc, tmp);
      }
      if constexpr (ZPA && ZPB) {
        tmp = _mm256_mullo_epi32(vzpb, vk);
        tmp = _mm256_mullo_epi32(tmp, vzpa);
        vacc = _mm256_sub_epi32(vacc, tmp);
      }
      _mm256_storeu_si256((__m256i*)&accptr[i * ldacc + j], vacc);
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
template <int NTILE>
static inline BTLA_CODE decompress_kblock_s4_s8_pack4_row(utils::int4x2* srcptr, int8_t* zpptr, int8_t* dstptr,
                                                          int blocksize, int ldzp, int n_offset, int k_offset, int row,
                                                          int8_t* tmp, size_t tmpsize) {
  int constexpr NReg = NTILE / 8;
  static_assert((NTILE % 8) == 0);
  int constexpr PackRow = 4;
  __m256i v_zp_y[NReg];
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm256_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm256_set1_epi8(8);
  const auto vindex = _mm256_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0);
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = load_zp_epi8_broadcast_epi32(zptr + i * 8, vindex);
      v_zp_y[i] = _mm256_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    for (int ib = 0; ib < k_remain; ib += PackRow) {
      auto b4ptr = srcptr + (ir + ib) * NTILE / 2;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_4bits(b4ptr + i * 16, vmask);
        v_s8_y = _mm256_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(dstptr + i * 32 + (ir + ib) * NTILE), v_s8_y);
      }
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s4_s8_pack2_row(utils::int4x2* srcptr, int8_t* zpptr, int8_t* dstptr,
                                                          int blocksize, int ldzp, int n_offset, int k_offset, int row,
                                                          int8_t* tmp, size_t tmpsize) {
  int constexpr NReg = NTILE / 8;
  static_assert((NTILE % 8) == 0);
  int constexpr PackRow = 2;
  int constexpr Unroll = 2;
  __m256i v_zp_y[NReg];
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm256_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm256_set1_epi8(8);
  const auto vindex = _mm256_set_epi8(14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0, 14, 14, 12, 12, 10, 10, 8,
                                      8, 6, 6, 4, 4, 2, 2, 0, 0);
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    memcpy(tmp, zptr, NTILE * sizeof(int8_t));
    memcpy(tmp + NTILE, zptr, NTILE * sizeof(int8_t));
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = load_zp_epi8_broadcast_epi16_v16(tmp + i * 16, vindex);
      v_zp_y[i] = _mm256_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    int k_remain_unrll = utils::padto_le(k_remain, PackRow * Unroll);
    int ib = 0;
    for (; ib < k_remain_unrll; ib += PackRow * Unroll) {
      auto b4ptr = srcptr + (ir + ib) * NTILE / 2;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_4bits(b4ptr + i * 16, vmask);
        v_s8_y = _mm256_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(dstptr + i * 32 + (ir + ib) * NTILE), v_s8_y);
      }
    }
    int k_tail = k_remain - k_remain_unrll;
    if (k_tail > 0) {
      memcpy(tmp, srcptr + (ir + ib) * NTILE / 2, k_tail * NTILE / 2);
      auto tmpout = tmp + Unroll * PackRow * NTILE / 2;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_4bits(tmp + i * 16, vmask);
        v_s8_y = _mm256_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(tmpout + i * 32), v_s8_y);
      }
      memcpy(dstptr + (ir + ib) * NTILE, tmpout, k_tail * NTILE);
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s4_s8_pack1_row(utils::int4x2* srcptr, int8_t* zpptr, int8_t* dstptr,
                                                          int blocksize, int ldzp, int n_offset, int k_offset, int row,
                                                          int8_t* tmp, size_t tmpsize) {
  int constexpr NReg = NTILE / 8;
  static_assert((NTILE % 8) == 0);
  int constexpr PackRow = 1;
  int constexpr Unroll = 4;
  int constexpr UnpackLoop = Unroll * NTILE / 32;
  __m256i v_zp_y[UnpackLoop];
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm256_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm256_set1_epi8(8);
  const auto vindex = _mm256_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0);
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    for (int i = 0; i < Unroll; i++) {
      memcpy(tmp + i * NTILE, zptr, NTILE * sizeof(int8_t));
    }
    for (int i = 0; i < UnpackLoop; i++) {
      v_zp_y[i] = _mm256_loadu_si256((const __m256i*)(tmp + i * 32));
      v_zp_y[i] = _mm256_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    int k_remain_unrll = utils::padto_le(k_remain, Unroll);
    int ib = 0;
    for (; ib < k_remain_unrll; ib += Unroll) {
      auto b4ptr = srcptr + (ir + ib) * NTILE / 2;
      for (int i = 0; i < UnpackLoop; i++) {
        auto v_s8_y = unpack_4bits(b4ptr + i * 16, vmask);
        v_s8_y = _mm256_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(dstptr + i * 32 + (ir + ib) * NTILE), v_s8_y);
      }
    }

    int k_tail = k_remain - k_remain_unrll;
    if (k_tail > 0) {
      memcpy(tmp, srcptr + (ir + ib) * NTILE / 2, k_tail * NTILE / 2);
      auto tmpout = tmp + Unroll * NTILE / 2;
      for (int i = 0; i < UnpackLoop; i++) {
        auto v_s8_y = unpack_4bits(tmp + i * 16, vmask);
        v_s8_y = _mm256_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(tmpout + i * 32), v_s8_y);
      }
      memcpy(dstptr + (ir + ib) * NTILE, tmpout, k_tail * NTILE);
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE decompress_s4_s8(utils::int4x2* srcptr, int8_t* dstptr, size_t elesize, int8_t* tmp,
                                         size_t tmpsize) {
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm256_set1_epi32(*reinterpret_cast<int*>(&mask));
  size_t velt = utils::padto_le(elesize, 32);
  size_t i = 0;
  auto vbias = _mm256_set1_epi8(8);
  for (; i < velt; i += 32) {
    auto vout_y = unpack_4bits(reinterpret_cast<int8_t*>(srcptr + i / 2), vmask);
    vout_y = _mm256_sub_epi8(vout_y, vbias);
    _mm256_storeu_si256((__m256i*)(dstptr + i), vout_y);
  }
  if (velt < elesize) {
    if (elesize >= 32) {
      i = elesize - 32;
      auto vout_y = unpack_4bits(reinterpret_cast<int8_t*>(srcptr + i / 2), vmask);
      vout_y = _mm256_sub_epi8(vout_y, vbias);
      _mm256_storeu_si256((__m256i*)(dstptr + i), vout_y);
    } else {
      ref::decompress_kblock_s4_s8<1, 1>(srcptr + i / 2, nullptr, dstptr + i, 0, 0, 0, 0, 1,
                                         static_cast<int>(elesize - i), nullptr, 0);
    }
  }
  return BTLA_CODE::Success;
}

template <int PackRow, int NTILE>
static inline BTLA_CODE decompress_kblock_s4_s8(utils::int4x2* srcptr, int8_t* zpptr, int8_t* dstptr, int blocksize,
                                                int ldzp, int n_offset, int k_offset, int row, int col, int8_t* tmp,
                                                size_t tmpsize) {
  if (zpptr) {
    typedef BTLA_CODE (*decompfunc)(utils::int4x2* srcptr, int8_t* zpptr, int8_t* dstptr, int blocksize, int ldzp,
                                    int n_offset, int k_offset, int row, int8_t* tmp, size_t tmpsize);
    decompfunc func = nullptr;
    if (col == NTILE) {
      if constexpr (PackRow == 4) {
        func = &decompress_kblock_s4_s8_pack4_row<NTILE>;
      }
      if constexpr (PackRow == 1) {
        func = &decompress_kblock_s4_s8_pack1_row<NTILE>;
      }
      if constexpr (PackRow == 2) {
        func = &decompress_kblock_s4_s8_pack2_row<NTILE>;
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

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s2_s8_pack4_row(utils::bit2x4* srcptr, int8_t* zpptr, int8_t* dstptr,
                                                          int blocksize, int ldzp, int n_offset, int k_offset, int row,
                                                          int8_t* tmp, size_t tmpsize) {
  int constexpr NReg = NTILE / 8;
  static_assert((NTILE % 8) == 0);
  int constexpr PackRow = 4;
  __m256i v_zp_y[NReg];
  const auto vindex = _mm256_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0);
  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm256_set_epi64x(*(int64_t*)&mask0, *(int64_t*)&mask0, *(int64_t*)&mask0, *(int64_t*)&mask0);
  auto vbias = _mm256_set1_epi8(2);
  auto vshift_y = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm256_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = load_zp_epi8_broadcast_epi32(zptr + i * 8, vindex);
      v_zp_y[i] = _mm256_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    for (int ib = 0; ib < k_remain; ib += PackRow) {
      auto b2ptr = srcptr + (ir + ib) * NTILE / 4;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_2bits(b2ptr + i * 8, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        v_s8_y = _mm256_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(dstptr + i * 32 + (ir + ib) * NTILE), v_s8_y);
      }
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s2_s8_pack2_row(utils::bit2x4* srcptr, int8_t* zpptr, int8_t* dstptr,
                                                          int blocksize, int ldzp, int n_offset, int k_offset, int row,
                                                          int8_t* tmp, size_t tmpsize) {
  int constexpr NReg = NTILE / 8;
  static_assert((NTILE % 8) == 0);
  int constexpr PackRow = 2;
  int constexpr Unroll = 2;
  __m256i v_zp_y[NReg];
  const auto vindex = _mm256_set_epi8(14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0, 14, 14, 12, 12, 10, 10, 8,
                                      8, 6, 6, 4, 4, 2, 2, 0, 0);
  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm256_set_epi64x(*(int64_t*)&mask0, *(int64_t*)&mask0, *(int64_t*)&mask0, *(int64_t*)&mask0);
  auto vbias = _mm256_set1_epi8(2);
  auto vshift_y = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm256_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    memcpy(tmp, zptr, NTILE * sizeof(int8_t));
    memcpy(tmp + NTILE, zptr, NTILE * sizeof(int8_t));
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = load_zp_epi8_broadcast_epi16_v16(tmp + i * 16, vindex);
      v_zp_y[i] = _mm256_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    int k_remain_unrll = utils::padto_le(k_remain, PackRow * Unroll);
    int ib = 0;
    for (; ib < k_remain_unrll; ib += PackRow * Unroll) {
      auto b2ptr = srcptr + (ir + ib) * NTILE / 4;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_2bits(b2ptr + i * 8, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        v_s8_y = _mm256_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(dstptr + i * 32 + (ir + ib) * NTILE), v_s8_y);
      }
    }
    int k_tail = k_remain - k_remain_unrll;
    if (k_tail > 0) {
      memcpy(tmp, srcptr + (ir + ib) * NTILE / 4, k_tail * NTILE / 4);
      auto tmpout = tmp + Unroll * PackRow * NTILE / 4;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_2bits((utils::bit2x4*)(tmp + i * 8), vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        v_s8_y = _mm256_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(tmpout + i * 32), v_s8_y);
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
  int constexpr NReg = NTILE / 8;
  static_assert((NTILE % 8) == 0);
  int constexpr PackRow = 1;
  int constexpr Unroll = 4;
  int constexpr UnpackLoop = Unroll * NTILE / 32;
  __m256i v_zp_y[UnpackLoop];
  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm256_set_epi64x(*(int64_t*)&mask0, *(int64_t*)&mask0, *(int64_t*)&mask0, *(int64_t*)&mask0);
  auto vbias = _mm256_set1_epi8(2);
  auto vshift_y = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm256_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    for (int i = 0; i < Unroll; i++) {
      memcpy(tmp + i * NTILE, zptr, NTILE * sizeof(int8_t));
    }
    for (int i = 0; i < UnpackLoop; i++) {
      v_zp_y[i] = _mm256_loadu_si256((const __m256i*)(tmp + i * 32));
      v_zp_y[i] = _mm256_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    int k_remain_unrll = utils::padto_le(k_remain, Unroll);
    int ib = 0;
    for (; ib < k_remain_unrll; ib += Unroll) {
      auto b2ptr = srcptr + (ir + ib) * NTILE / 4;
      for (int i = 0; i < UnpackLoop; i++) {
        auto v_s8_y = unpack_2bits(b2ptr + i * 8, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        v_s8_y = _mm256_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(dstptr + i * 32 + (ir + ib) * NTILE), v_s8_y);
      }
    }

    int k_tail = k_remain - k_remain_unrll;
    if (k_tail > 0) {
      memcpy(tmp, srcptr + (ir + ib) * NTILE / 4, k_tail * NTILE / 4);
      auto tmpout = tmp + Unroll * NTILE / 4;
      for (int i = 0; i < UnpackLoop; i++) {
        auto v_s8_y = unpack_2bits((utils::bit2x4*)(tmp + i * 8), vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        v_s8_y = _mm256_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(tmpout + i * 32), v_s8_y);
      }
      memcpy(dstptr + (ir + ib) * NTILE, tmpout, k_tail * NTILE);
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE decompress_s2_s8(utils::bit2x4* bit2ptr, int8_t* dstptr, size_t unpack_elt, int8_t* tmp,
                                         size_t tmpsize) {
  int constexpr VBits = 256;
  int constexpr VElt = VBits / 8;
  size_t i = 0;
  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm256_set_epi64x(*(int64_t*)&mask0, *(int64_t*)&mask0, *(int64_t*)&mask0, *(int64_t*)&mask0);
  auto vbias = _mm256_set1_epi8(2);
  auto vshift_y = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm256_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);
  size_t elt_pad = utils::padto_le(unpack_elt, VElt);
  for (; i < elt_pad; i += VElt) {
    auto vout = unpack_2bits(bit2ptr + i / 4, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
    vout = _mm256_sub_epi8(vout, vbias);
    _mm256_storeu_si256((__m256i*)(dstptr + i), vout);
  }
  if (elt_pad < unpack_elt) {
    if (unpack_elt >= 32) {
      i = unpack_elt - 32;
      auto vout = unpack_2bits(bit2ptr + i / 4, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
      vout = _mm256_sub_epi8(vout, vbias);
      _mm256_storeu_si256((__m256i*)(dstptr + i), vout);
    } else {
      ref::decompress_s2_s8(bit2ptr + i / 4, dstptr + i, unpack_elt - i, tmp, tmpsize);
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
      if constexpr (PackRow == 4) {
        func = &decompress_kblock_s2_s8_pack4_row<NTILE>;
      }
      if constexpr (PackRow == 1) {
        func = &decompress_kblock_s2_s8_pack1_row<NTILE>;
      }
      if constexpr (PackRow == 2) {
        func = &decompress_kblock_s2_s8_pack2_row<NTILE>;
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

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s3_s8_pack4_row(utils::bit2x4* srcptr, utils::bit1x8* bit1ptr, int8_t* zpptr,
                                                          int8_t* dstptr, int blocksize, int ldzp, int n_offset,
                                                          int k_offset, int row, int8_t* tmp, size_t tmpsize) {
  int constexpr NReg = NTILE / 8;
  static_assert((NTILE % 8) == 0);
  int constexpr PackRow = 4;
  __m256i v_zp_y[NReg];
  const auto vindex = _mm256_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0);
  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm256_set_epi64x(*(int64_t*)&mask0, *(int64_t*)&mask0, *(int64_t*)&mask0, *(int64_t*)&mask0);
  auto vbias = _mm256_set1_epi8(4);
  auto vshift_y = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm256_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);

  const __m256i highMask = _mm256_set1_epi8(0x04);
  const __m256i bit1Mask = _mm256_set1_epi32(0x0F);
  const __m256i bit1Shift_1 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
  const __m256i bit1Shift_2 = _mm256_set1_epi32((1 << 23) + (1 << 16) + (1 << 9) + (1 << 2));
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = load_zp_epi8_broadcast_epi32(zptr + i * 8, vindex);
      v_zp_y[i] = _mm256_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    for (int ib = 0; ib < k_remain; ib += PackRow) {
      auto b2ptr = srcptr + (ir + ib) * NTILE / 4;
      auto b1ptr = bit1ptr + (ir + ib) * NTILE / 8;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_2bits(b2ptr + i * 8, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        auto vb1 = unpack_1bits(b1ptr + i * 4, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
        v_s8_y = _mm256_or_si256(v_s8_y, vb1);
        v_s8_y = _mm256_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(dstptr + i * 32 + (ir + ib) * NTILE), v_s8_y);
      }
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s3_s8_pack2_row(utils::bit2x4* srcptr, utils::bit1x8* bit1ptr, int8_t* zpptr,
                                                          int8_t* dstptr, int blocksize, int ldzp, int n_offset,
                                                          int k_offset, int row, int8_t* tmp, size_t tmpsize) {
  int constexpr NReg = NTILE / 8;
  static_assert((NTILE % 8) == 0);
  int constexpr PackRow = 2;
  int constexpr Unroll = 2;
  __m256i v_zp_y[NReg];
  const auto vindex = _mm256_set_epi8(14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0, 14, 14, 12, 12, 10, 10, 8,
                                      8, 6, 6, 4, 4, 2, 2, 0, 0);
  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm256_set_epi64x(*(int64_t*)&mask0, *(int64_t*)&mask0, *(int64_t*)&mask0, *(int64_t*)&mask0);
  auto vbias = _mm256_set1_epi8(4);
  auto vshift_y = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm256_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);

  const __m256i highMask = _mm256_set1_epi8(0x04);
  const __m256i bit1Mask = _mm256_set1_epi32(0x0F);
  const __m256i bit1Shift_1 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
  const __m256i bit1Shift_2 = _mm256_set1_epi32((1 << 23) + (1 << 16) + (1 << 9) + (1 << 2));

  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    memcpy(tmp, zptr, NTILE * sizeof(int8_t));
    memcpy(tmp + NTILE, zptr, NTILE * sizeof(int8_t));
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = load_zp_epi8_broadcast_epi16_v16(tmp + i * 16, vindex);
      v_zp_y[i] = _mm256_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    int k_remain_unrll = utils::padto_le(k_remain, PackRow * Unroll);
    int ib = 0;
    for (; ib < k_remain_unrll; ib += PackRow * Unroll) {
      auto b2ptr = srcptr + (ir + ib) * NTILE / 4;
      auto b1ptr = bit1ptr + (ir + ib) * NTILE / 8;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_2bits(b2ptr + i * 8, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        auto vb1 = unpack_1bits(b1ptr + i * 4, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
        v_s8_y = _mm256_or_si256(v_s8_y, vb1);
        v_s8_y = _mm256_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(dstptr + i * 32 + (ir + ib) * NTILE), v_s8_y);
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
        auto v_s8_y = unpack_2bits((utils::bit2x4*)(tmpb2ptr + i * 8), vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        auto vb1 = unpack_1bits((utils::bit1x8*)(tmpb1ptr + i * 4), bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
        v_s8_y = _mm256_or_si256(v_s8_y, vb1);
        v_s8_y = _mm256_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(tmpout + i * 32), v_s8_y);
      }
      memcpy(dstptr + (ir + ib) * NTILE, tmpout, k_tail * NTILE);
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s3_s8_pack1_row(utils::bit2x4* srcptr, utils::bit1x8* bit1ptr, int8_t* zpptr,
                                                          int8_t* dstptr, int blocksize, int ldzp, int n_offset,
                                                          int k_offset, int row, int8_t* tmp, size_t tmpsize) {
  int constexpr NReg = NTILE / 8;
  static_assert((NTILE % 8) == 0);
  int constexpr PackRow = 1;
  int constexpr Unroll = 4;
  int constexpr UnpackLoop = Unroll * NTILE / 32;
  __m256i v_zp_y[UnpackLoop];
  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm256_set_epi64x(*(int64_t*)&mask0, *(int64_t*)&mask0, *(int64_t*)&mask0, *(int64_t*)&mask0);
  auto vbias = _mm256_set1_epi8(4);
  auto vshift_y = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm256_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);

  const __m256i highMask = _mm256_set1_epi8(0x04);
  const __m256i bit1Mask = _mm256_set1_epi32(0x0F);
  const __m256i bit1Shift_1 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
  const __m256i bit1Shift_2 = _mm256_set1_epi32((1 << 23) + (1 << 16) + (1 << 9) + (1 << 2));
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    for (int i = 0; i < Unroll; i++) {
      memcpy(tmp + i * NTILE, zptr, NTILE * sizeof(int8_t));
    }
    for (int i = 0; i < UnpackLoop; i++) {
      v_zp_y[i] = _mm256_loadu_si256((const __m256i*)(tmp + i * 32));
      v_zp_y[i] = _mm256_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    int k_remain_unrll = utils::padto_le(k_remain, Unroll);
    int ib = 0;
    for (; ib < k_remain_unrll; ib += Unroll) {
      auto b2ptr = srcptr + (ir + ib) * NTILE / 4;
      auto b1ptr = bit1ptr + (ir + ib) * NTILE / 8;
      for (int i = 0; i < UnpackLoop; i++) {
        auto v_s8_y = unpack_2bits(b2ptr + i * 8, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        auto vb1 = unpack_1bits(b1ptr + i * 4, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
        v_s8_y = _mm256_or_si256(v_s8_y, vb1);
        v_s8_y = _mm256_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(dstptr + i * 32 + (ir + ib) * NTILE), v_s8_y);
      }
    }

    int k_tail = k_remain - k_remain_unrll;
    if (k_tail > 0) {
      auto tmpb2ptr = tmp;
      memcpy(tmpb2ptr, srcptr + (ir + ib) * NTILE / 4, k_tail * NTILE / 4);
      auto tmpb1ptr = tmp + Unroll * NTILE / 2;
      memcpy(tmpb1ptr, bit1ptr + (ir + ib) * NTILE / 8, k_tail * NTILE / 8);
      auto tmpout = tmp + Unroll * NTILE;
      for (int i = 0; i < UnpackLoop; i++) {
        auto v_s8_y = unpack_2bits((utils::bit2x4*)(tmpb2ptr + i * 8), vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        auto vb1 = unpack_1bits((utils::bit1x8*)(tmpb1ptr + i * 4), bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
        v_s8_y = _mm256_or_si256(v_s8_y, vb1);
        v_s8_y = _mm256_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(tmpout + i * 32), v_s8_y);
      }
      memcpy(dstptr + (ir + ib) * NTILE, tmpout, k_tail * NTILE);
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE decompress_s3_s8(utils::bit2x4* bit2ptr, utils::bit1x8* bit1ptr, int8_t* dstptr,
                                         size_t unpack_elt, int8_t* tmp, size_t tmpsize) {
  int constexpr VBits = 256;
  int constexpr VElt = VBits / 8;
  size_t i = 0;
  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm256_set_epi64x(*(int64_t*)&mask0, *(int64_t*)&mask0, *(int64_t*)&mask0, *(int64_t*)&mask0);
  auto vbias = _mm256_set1_epi8(4);
  auto vshift_y = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm256_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);

  const __m256i highMask = _mm256_set1_epi8(0x04);
  const __m256i bit1Mask = _mm256_set1_epi32(0x0F);
  const __m256i bit1Shift_1 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
  const __m256i bit1Shift_2 = _mm256_set1_epi32((1 << 23) + (1 << 16) + (1 << 9) + (1 << 2));
  size_t elt_pad = utils::padto_le(unpack_elt, VElt);
  for (; i < elt_pad; i += VElt) {
    auto vout = unpack_2bits(bit2ptr + i / 4, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
    auto vb1 = unpack_1bits(bit1ptr + i / 8, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
    vout = _mm256_or_si256(vout, vb1);
    vout = _mm256_sub_epi8(vout, vbias);
    _mm256_storeu_si256((__m256i*)(dstptr + i), vout);
  }
  if (elt_pad < unpack_elt) {
    if (unpack_elt >= 32) {
      i = unpack_elt - 32;
      auto vout = unpack_2bits(bit2ptr + i / 4, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
      auto vb1 = unpack_1bits(bit1ptr + i / 8, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
      vout = _mm256_or_si256(vout, vb1);
      vout = _mm256_sub_epi8(vout, vbias);
      _mm256_storeu_si256((__m256i*)(dstptr + i), vout);
    } else {
      ref::decompress_s3_s8(bit2ptr + i / 4, bit1ptr + i / 8, dstptr + i, unpack_elt - i, tmp, tmpsize);
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

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s1_s8_pack4_row(utils::bit1x8* bit1ptr, int8_t* zpptr, int8_t* dstptr,
                                                          int blocksize, int ldzp, int n_offset, int k_offset, int row,
                                                          int8_t* tmp, size_t tmpsize) {
  int constexpr NReg = NTILE / 8;
  static_assert((NTILE % 8) == 0);
  int constexpr PackRow = 4;
  __m256i v_zp_y[NReg];
  const auto vindex = _mm256_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0);
  int constexpr FullRange = 1 << (1 - 1);
  auto vbias = _mm256_set1_epi8(FullRange);

  const __m256i highMask = _mm256_set1_epi8(0x04);
  const __m256i bit1Mask = _mm256_set1_epi32(0x0F);
  const __m256i bit1Shift_1 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
  const __m256i bit1Shift_2 = _mm256_set1_epi32((1 << 23) + (1 << 16) + (1 << 9) + (1 << 2));
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = load_zp_epi8_broadcast_epi32(zptr + i * 8, vindex);
      v_zp_y[i] = _mm256_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    for (int ib = 0; ib < k_remain; ib += PackRow) {
      auto b1ptr = bit1ptr + (ir + ib) * NTILE / 8;
      for (int i = 0; i < NReg; i++) {
        auto vb1 = unpack_1bits(b1ptr + i * 4, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
        vb1 = _mm256_srli_epi32(vb1, 2);
        vb1 = _mm256_sub_epi8(vb1, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(dstptr + i * 32 + (ir + ib) * NTILE), vb1);
      }
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s1_s8_pack2_row(utils::bit1x8* bit1ptr, int8_t* zpptr, int8_t* dstptr,
                                                          int blocksize, int ldzp, int n_offset, int k_offset, int row,
                                                          int8_t* tmp, size_t tmpsize) {
  int constexpr NReg = NTILE / 8;
  static_assert((NTILE % 8) == 0);
  int constexpr PackRow = 2;
  int constexpr Unroll = 2;
  __m256i v_zp_y[NReg];
  const auto vindex = _mm256_set_epi8(14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0, 14, 14, 12, 12, 10, 10, 8,
                                      8, 6, 6, 4, 4, 2, 2, 0, 0);
  int constexpr FullRange = 1 << (1 - 1);
  auto vbias = _mm256_set1_epi8(FullRange);

  const __m256i highMask = _mm256_set1_epi8(0x04);
  const __m256i bit1Mask = _mm256_set1_epi32(0x0F);
  const __m256i bit1Shift_1 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
  const __m256i bit1Shift_2 = _mm256_set1_epi32((1 << 23) + (1 << 16) + (1 << 9) + (1 << 2));

  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    memcpy(tmp, zptr, NTILE * sizeof(int8_t));
    memcpy(tmp + NTILE, zptr, NTILE * sizeof(int8_t));
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = load_zp_epi8_broadcast_epi16_v16(tmp + i * 16, vindex);
      v_zp_y[i] = _mm256_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    int k_remain_unrll = utils::padto_le(k_remain, PackRow * Unroll);
    int ib = 0;
    for (; ib < k_remain_unrll; ib += PackRow * Unroll) {
      auto b1ptr = bit1ptr + (ir + ib) * NTILE / 8;
      for (int i = 0; i < NReg; i++) {
        auto vb1 = unpack_1bits(b1ptr + i * 4, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
        vb1 = _mm256_srli_epi32(vb1, 2);
        vb1 = _mm256_sub_epi8(vb1, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(dstptr + i * 32 + (ir + ib) * NTILE), vb1);
      }
    }
    int k_tail = k_remain - k_remain_unrll;
    if (k_tail > 0) {
      auto tmpb1ptr = tmp;
      memcpy(tmpb1ptr, bit1ptr + (ir + ib) * NTILE / 8, k_tail * NTILE / 8);
      auto tmpout = tmp + Unroll * NTILE;
      for (int i = 0; i < NReg; i++) {
        auto vb1 = unpack_1bits((utils::bit1x8*)(tmpb1ptr + i * 4), bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
        vb1 = _mm256_srli_epi32(vb1, 2);
        vb1 = _mm256_sub_epi8(vb1, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(tmpout + i * 32), vb1);
      }
      memcpy(dstptr + (ir + ib) * NTILE, tmpout, k_tail * NTILE);
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s1_s8_pack1_row(utils::bit1x8* bit1ptr, int8_t* zpptr, int8_t* dstptr,
                                                          int blocksize, int ldzp, int n_offset, int k_offset, int row,
                                                          int8_t* tmp, size_t tmpsize) {
  int constexpr NReg = NTILE / 8;
  static_assert((NTILE % 8) == 0);
  int constexpr PackRow = 1;
  int constexpr Unroll = 4;
  int constexpr UnpackLoop = Unroll * NTILE / 32;
  int constexpr FullRange = 1 << (1 - 1);
  auto vbias = _mm256_set1_epi8(FullRange);
  __m256i v_zp_y[UnpackLoop];

  const __m256i highMask = _mm256_set1_epi8(0x04);
  const __m256i bit1Mask = _mm256_set1_epi32(0x0F);
  const __m256i bit1Shift_1 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
  const __m256i bit1Shift_2 = _mm256_set1_epi32((1 << 23) + (1 << 16) + (1 << 9) + (1 << 2));
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    for (int i = 0; i < Unroll; i++) {
      memcpy(tmp + i * NTILE, zptr, NTILE * sizeof(int8_t));
    }
    for (int i = 0; i < UnpackLoop; i++) {
      v_zp_y[i] = _mm256_loadu_si256((const __m256i*)(tmp + i * 32));
      v_zp_y[i] = _mm256_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    int k_remain_unrll = utils::padto_le(k_remain, Unroll);
    int ib = 0;
    for (; ib < k_remain_unrll; ib += Unroll) {
      auto b1ptr = bit1ptr + (ir + ib) * NTILE / 8;
      for (int i = 0; i < UnpackLoop; i++) {
        auto vb1 = unpack_1bits(b1ptr + i * 4, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
        vb1 = _mm256_srli_epi32(vb1, 2);
        vb1 = _mm256_sub_epi8(vb1, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(dstptr + i * 32 + (ir + ib) * NTILE), vb1);
      }
    }

    int k_tail = k_remain - k_remain_unrll;
    if (k_tail > 0) {
      auto tmpb1ptr = tmp;
      memcpy(tmpb1ptr, bit1ptr + (ir + ib) * NTILE / 8, k_tail * NTILE / 8);
      auto tmpout = tmp + Unroll * NTILE;
      for (int i = 0; i < UnpackLoop; i++) {
        auto vb1 = unpack_1bits((utils::bit1x8*)(tmpb1ptr + i * 4), bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
        vb1 = _mm256_srli_epi32(vb1, 2);
        vb1 = _mm256_sub_epi8(vb1, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(tmpout + i * 32), vb1);
      }
      memcpy(dstptr + (ir + ib) * NTILE, tmpout, k_tail * NTILE);
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE decompress_s1_s8(utils::bit1x8* bit1ptr, int8_t* dstptr, size_t unpack_elt, int8_t* tmp,
                                         size_t tmpsize) {
  int constexpr VBits = 256;
  int constexpr VElt = VBits / 8;
  size_t i = 0;
  int constexpr FullRange = 1 << (1 - 1);
  auto vbias = _mm256_set1_epi8(FullRange);

  const __m256i highMask = _mm256_set1_epi8(0x04);
  const __m256i bit1Mask = _mm256_set1_epi32(0x0F);
  const __m256i bit1Shift_1 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
  const __m256i bit1Shift_2 = _mm256_set1_epi32((1 << 23) + (1 << 16) + (1 << 9) + (1 << 2));
  size_t elt_pad = utils::padto_le(unpack_elt, VElt);
  for (; i < elt_pad; i += VElt) {
    auto vb1 = unpack_1bits(bit1ptr + i / 8, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
    vb1 = _mm256_srli_epi32(vb1, 2);
    vb1 = _mm256_sub_epi8(vb1, vbias);
    _mm256_storeu_si256((__m256i*)(dstptr + i), vb1);
  }
  if (elt_pad < unpack_elt) {
    if (unpack_elt >= 32) {
      i = unpack_elt - 32;
      auto vb1 = unpack_1bits(bit1ptr + i / 8, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
      vb1 = _mm256_srli_epi32(vb1, 2);
      vb1 = _mm256_sub_epi8(vb1, vbias);
      _mm256_storeu_si256((__m256i*)(dstptr + i), vb1);
    } else {
      ref::decompress_s1_s8(bit1ptr + i / 8, dstptr + i, unpack_elt - i, tmp, tmpsize);
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

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s5_s8_pack4_row(utils::bit4x2* srcptr, utils::bit1x8* bit1ptr, int8_t* zpptr,
                                                          int8_t* dstptr, int blocksize, int ldzp, int n_offset,
                                                          int k_offset, int row, int8_t* tmp, size_t tmpsize) {
  int constexpr NReg = NTILE / 8;
  static_assert((NTILE % 8) == 0);
  int constexpr PackRow = 4;
  __m256i v_zp_y[NReg];
  const auto vindex = _mm256_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0);
  int constexpr FullRange = 1 << (5 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm256_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm256_set1_epi8(FullRange);

  const __m256i highMask = _mm256_set1_epi8(0x04);
  const __m256i bit1Mask = _mm256_set1_epi32(0x0F);
  const __m256i bit1Shift_1 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
  const __m256i bit1Shift_2 = _mm256_set1_epi32((1 << 23) + (1 << 16) + (1 << 9) + (1 << 2));
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = load_zp_epi8_broadcast_epi32(zptr + i * 8, vindex);
      v_zp_y[i] = _mm256_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    for (int ib = 0; ib < k_remain; ib += PackRow) {
      auto b4ptr = srcptr + (ir + ib) * NTILE / 2;
      auto b1ptr = bit1ptr + (ir + ib) * NTILE / 8;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_4bits(b4ptr + i * 16, vmask);
        auto vb1 = unpack_1bits(b1ptr + i * 4, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
        vb1 = _mm256_slli_epi32(vb1, 2);
        v_s8_y = _mm256_or_si256(v_s8_y, vb1);
        v_s8_y = _mm256_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(dstptr + i * 32 + (ir + ib) * NTILE), v_s8_y);
      }
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s5_s8_pack2_row(utils::bit4x2* srcptr, utils::bit1x8* bit1ptr, int8_t* zpptr,
                                                          int8_t* dstptr, int blocksize, int ldzp, int n_offset,
                                                          int k_offset, int row, int8_t* tmp, size_t tmpsize) {
  int constexpr NReg = NTILE / 8;
  static_assert((NTILE % 8) == 0);
  int constexpr PackRow = 2;
  int constexpr Unroll = 2;
  __m256i v_zp_y[NReg];
  const auto vindex = _mm256_set_epi8(14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0, 14, 14, 12, 12, 10, 10, 8,
                                      8, 6, 6, 4, 4, 2, 2, 0, 0);
  int constexpr FullRange = 1 << (5 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm256_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm256_set1_epi8(FullRange);

  const __m256i highMask = _mm256_set1_epi8(0x04);
  const __m256i bit1Mask = _mm256_set1_epi32(0x0F);
  const __m256i bit1Shift_1 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
  const __m256i bit1Shift_2 = _mm256_set1_epi32((1 << 23) + (1 << 16) + (1 << 9) + (1 << 2));

  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    memcpy(tmp, zptr, NTILE * sizeof(int8_t));
    memcpy(tmp + NTILE, zptr, NTILE * sizeof(int8_t));
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = load_zp_epi8_broadcast_epi16_v16(tmp + i * 16, vindex);
      v_zp_y[i] = _mm256_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    int k_remain_unrll = utils::padto_le(k_remain, PackRow * Unroll);
    int ib = 0;
    for (; ib < k_remain_unrll; ib += PackRow * Unroll) {
      auto b4ptr = srcptr + (ir + ib) * NTILE / 2;
      auto b1ptr = bit1ptr + (ir + ib) * NTILE / 8;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_4bits(b4ptr + i * 16, vmask);
        auto vb1 = unpack_1bits(b1ptr + i * 4, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
        vb1 = _mm256_slli_epi32(vb1, 2);
        v_s8_y = _mm256_or_si256(v_s8_y, vb1);
        v_s8_y = _mm256_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(dstptr + i * 32 + (ir + ib) * NTILE), v_s8_y);
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
        auto v_s8_y = unpack_4bits((utils::bit4x2*)(tmpb4ptr + i * 16), vmask);
        auto vb1 = unpack_1bits((utils::bit1x8*)(tmpb1ptr + i * 4), bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
        vb1 = _mm256_slli_epi32(vb1, 2);
        v_s8_y = _mm256_or_si256(v_s8_y, vb1);
        v_s8_y = _mm256_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(tmpout + i * 32), v_s8_y);
      }
      memcpy(dstptr + (ir + ib) * NTILE, tmpout, k_tail * NTILE);
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s5_s8_pack1_row(utils::bit4x2* srcptr, utils::bit1x8* bit1ptr, int8_t* zpptr,
                                                          int8_t* dstptr, int blocksize, int ldzp, int n_offset,
                                                          int k_offset, int row, int8_t* tmp, size_t tmpsize) {
  int constexpr NReg = NTILE / 8;
  static_assert((NTILE % 8) == 0);
  int constexpr PackRow = 1;
  int constexpr Unroll = 4;
  int constexpr UnpackLoop = Unroll * NTILE / 32;
  int constexpr FullRange = 1 << (5 - 1);
  __m256i v_zp_y[UnpackLoop];
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm256_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm256_set1_epi8(FullRange);

  const __m256i highMask = _mm256_set1_epi8(0x04);
  const __m256i bit1Mask = _mm256_set1_epi32(0x0F);
  const __m256i bit1Shift_1 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
  const __m256i bit1Shift_2 = _mm256_set1_epi32((1 << 23) + (1 << 16) + (1 << 9) + (1 << 2));
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    for (int i = 0; i < Unroll; i++) {
      memcpy(tmp + i * NTILE, zptr, NTILE * sizeof(int8_t));
    }
    for (int i = 0; i < UnpackLoop; i++) {
      v_zp_y[i] = _mm256_loadu_si256((const __m256i*)(tmp + i * 32));
      v_zp_y[i] = _mm256_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    int k_remain_unrll = utils::padto_le(k_remain, Unroll);
    int ib = 0;
    for (; ib < k_remain_unrll; ib += Unroll) {
      auto b4ptr = srcptr + (ir + ib) * NTILE / 2;
      auto b1ptr = bit1ptr + (ir + ib) * NTILE / 8;
      for (int i = 0; i < UnpackLoop; i++) {
        auto v_s8_y = unpack_4bits(b4ptr + i * 16, vmask);
        auto vb1 = unpack_1bits(b1ptr + i * 4, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
        vb1 = _mm256_slli_epi32(vb1, 2);
        v_s8_y = _mm256_or_si256(v_s8_y, vb1);
        v_s8_y = _mm256_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(dstptr + i * 32 + (ir + ib) * NTILE), v_s8_y);
      }
    }

    int k_tail = k_remain - k_remain_unrll;
    if (k_tail > 0) {
      auto tmpb4ptr = tmp;
      memcpy(tmpb4ptr, srcptr + (ir + ib) * NTILE / 2, k_tail * NTILE / 2);
      auto tmpb1ptr = tmp + Unroll * NTILE / 2;
      memcpy(tmpb1ptr, bit1ptr + (ir + ib) * NTILE / 8, k_tail * NTILE / 8);
      auto tmpout = tmp + Unroll * NTILE;
      for (int i = 0; i < UnpackLoop; i++) {
        auto v_s8_y = unpack_4bits((utils::bit4x2*)(tmpb4ptr + i * 16), vmask);
        auto vb1 = unpack_1bits((utils::bit1x8*)(tmpb1ptr + i * 4), bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
        vb1 = _mm256_slli_epi32(vb1, 2);
        v_s8_y = _mm256_or_si256(v_s8_y, vb1);
        v_s8_y = _mm256_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(tmpout + i * 32), v_s8_y);
      }
      memcpy(dstptr + (ir + ib) * NTILE, tmpout, k_tail * NTILE);
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE decompress_s5_s8(utils::bit4x2* bit4ptr, utils::bit1x8* bit1ptr, int8_t* dstptr,
                                         size_t unpack_elt, int8_t* tmp, size_t tmpsize) {
  int constexpr VBits = 256;
  int constexpr VElt = VBits / 8;
  size_t i = 0;
  int constexpr FullRange = 1 << (5 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm256_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm256_set1_epi8(FullRange);

  const __m256i highMask = _mm256_set1_epi8(0x04);
  const __m256i bit1Mask = _mm256_set1_epi32(0x0F);
  const __m256i bit1Shift_1 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
  const __m256i bit1Shift_2 = _mm256_set1_epi32((1 << 23) + (1 << 16) + (1 << 9) + (1 << 2));
  size_t elt_pad = utils::padto_le(unpack_elt, VElt);
  for (; i < elt_pad; i += VElt) {
    auto vout = unpack_4bits(bit4ptr + i / 2, vmask);
    auto vb1 = unpack_1bits(bit1ptr + i / 8, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
    vb1 = _mm256_slli_epi32(vb1, 2);
    vout = _mm256_or_si256(vout, vb1);
    vout = _mm256_sub_epi8(vout, vbias);
    _mm256_storeu_si256((__m256i*)(dstptr + i), vout);
  }
  if (elt_pad < unpack_elt) {
    if (unpack_elt >= 32) {
      i = unpack_elt - 32;
      auto vout = unpack_4bits(bit4ptr + i / 2, vmask);
      auto vb1 = unpack_1bits(bit1ptr + i / 8, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
      vb1 = _mm256_slli_epi32(vb1, 2);
      vout = _mm256_or_si256(vout, vb1);
      vout = _mm256_sub_epi8(vout, vbias);
      _mm256_storeu_si256((__m256i*)(dstptr + i), vout);
    } else {
      ref::decompress_s5_s8(bit4ptr + i / 2, bit1ptr + i / 8, dstptr + i, unpack_elt - i, tmp, tmpsize);
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

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s7_s8_pack4_row(utils::bit4x2* srcptr, utils::bit2x4* bit2ptr,
                                                          utils::bit1x8* bit1ptr, int8_t* zpptr, int8_t* dstptr,
                                                          int blocksize, int ldzp, int n_offset, int k_offset, int row,
                                                          int8_t* tmp, size_t tmpsize) {
  int constexpr NReg = NTILE / 8;
  static_assert((NTILE % 8) == 0);
  int constexpr PackRow = 4;
  __m256i v_zp_y[NReg];
  const auto vindex = _mm256_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0);
  int constexpr FullRange = 1 << (7 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm256_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm256_set1_epi8(FullRange);

  uint32_t mask0 = 0x03030303;
  auto vmask0 = _mm256_set1_epi32(*(int32_t*)&mask0);
  auto vshift_y = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm256_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);

  const __m256i highMask = _mm256_set1_epi8(0x04);
  const __m256i bit1Mask = _mm256_set1_epi32(0x0F);
  const __m256i bit1Shift_1 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
  const __m256i bit1Shift_2 = _mm256_set1_epi32((1 << 23) + (1 << 16) + (1 << 9) + (1 << 2));
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = load_zp_epi8_broadcast_epi32(zptr + i * 8, vindex);
      v_zp_y[i] = _mm256_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    for (int ib = 0; ib < k_remain; ib += PackRow) {
      auto b4ptr = srcptr + (ir + ib) * NTILE / 2;
      auto b1ptr = bit1ptr + (ir + ib) * NTILE / 8;
      auto b2ptr = bit2ptr + (ir + ib) * NTILE / 4;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_4bits(b4ptr + i * 16, vmask);
        auto vb1 = unpack_1bits(b1ptr + i * 4, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
        auto vb2 = unpack_2bits(b2ptr + i * 8, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        vb1 = _mm256_slli_epi32(vb1, 4);
        vb2 = _mm256_slli_epi32(vb2, 4);
        v_s8_y = _mm256_or_si256(v_s8_y, vb1);
        v_s8_y = _mm256_or_si256(v_s8_y, vb2);
        v_s8_y = _mm256_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(dstptr + i * 32 + (ir + ib) * NTILE), v_s8_y);
      }
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s7_s8_pack2_row(utils::bit4x2* srcptr, utils::bit2x4* bit2ptr,
                                                          utils::bit1x8* bit1ptr, int8_t* zpptr, int8_t* dstptr,
                                                          int blocksize, int ldzp, int n_offset, int k_offset, int row,
                                                          int8_t* tmp, size_t tmpsize) {
  int constexpr NReg = NTILE / 8;
  static_assert((NTILE % 8) == 0);
  int constexpr PackRow = 2;
  int constexpr Unroll = 2;
  __m256i v_zp_y[NReg];
  const auto vindex = _mm256_set_epi8(14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0, 14, 14, 12, 12, 10, 10, 8,
                                      8, 6, 6, 4, 4, 2, 2, 0, 0);
  int constexpr FullRange = 1 << (7 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm256_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm256_set1_epi8(FullRange);

  uint32_t mask0 = 0x03030303;
  auto vmask0 = _mm256_set1_epi32(*(int32_t*)&mask0);
  auto vshift_y = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm256_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);

  const __m256i highMask = _mm256_set1_epi8(0x04);
  const __m256i bit1Mask = _mm256_set1_epi32(0x0F);
  const __m256i bit1Shift_1 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
  const __m256i bit1Shift_2 = _mm256_set1_epi32((1 << 23) + (1 << 16) + (1 << 9) + (1 << 2));

  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    memcpy(tmp, zptr, NTILE * sizeof(int8_t));
    memcpy(tmp + NTILE, zptr, NTILE * sizeof(int8_t));
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = load_zp_epi8_broadcast_epi16_v16(tmp + i * 16, vindex);
      v_zp_y[i] = _mm256_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    int k_remain_unrll = utils::padto_le(k_remain, PackRow * Unroll);
    int ib = 0;
    for (; ib < k_remain_unrll; ib += PackRow * Unroll) {
      auto b4ptr = srcptr + (ir + ib) * NTILE / 2;
      auto b1ptr = bit1ptr + (ir + ib) * NTILE / 8;
      auto b2ptr = bit2ptr + (ir + ib) * NTILE / 4;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_4bits(b4ptr + i * 16, vmask);
        auto vb1 = unpack_1bits(b1ptr + i * 4, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
        auto vb2 = unpack_2bits(b2ptr + i * 8, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        vb1 = _mm256_slli_epi32(vb1, 4);
        vb2 = _mm256_slli_epi32(vb2, 4);
        v_s8_y = _mm256_or_si256(v_s8_y, vb1);
        v_s8_y = _mm256_or_si256(v_s8_y, vb2);
        v_s8_y = _mm256_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(dstptr + i * 32 + (ir + ib) * NTILE), v_s8_y);
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
        auto v_s8_y = unpack_4bits((utils::bit4x2*)(tmpb4ptr + i * 16), vmask);
        auto vb1 = unpack_1bits((utils::bit1x8*)(tmpb1ptr + i * 4), bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
        auto vb2 = unpack_2bits((utils::bit2x4*)(tmpb2ptr + i * 8), vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        vb1 = _mm256_slli_epi32(vb1, 4);
        vb2 = _mm256_slli_epi32(vb2, 4);
        v_s8_y = _mm256_or_si256(v_s8_y, vb1);
        v_s8_y = _mm256_or_si256(v_s8_y, vb2);
        v_s8_y = _mm256_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(tmpout + i * 32), v_s8_y);
      }
      memcpy(dstptr + (ir + ib) * NTILE, tmpout, k_tail * NTILE);
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s7_s8_pack1_row(utils::bit4x2* srcptr, utils::bit2x4* bit2ptr,
                                                          utils::bit1x8* bit1ptr, int8_t* zpptr, int8_t* dstptr,
                                                          int blocksize, int ldzp, int n_offset, int k_offset, int row,
                                                          int8_t* tmp, size_t tmpsize) {
  int constexpr NReg = NTILE / 8;
  static_assert((NTILE % 8) == 0);
  int constexpr PackRow = 1;
  int constexpr Unroll = 4;
  int constexpr UnpackLoop = Unroll * NTILE / 32;
  int constexpr FullRange = 1 << (7 - 1);
  __m256i v_zp_y[UnpackLoop];
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm256_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm256_set1_epi8(FullRange);

  uint32_t mask0 = 0x03030303;
  auto vmask0 = _mm256_set1_epi32(*(int32_t*)&mask0);
  auto vshift_y = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm256_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);

  const __m256i highMask = _mm256_set1_epi8(0x04);
  const __m256i bit1Mask = _mm256_set1_epi32(0x0F);
  const __m256i bit1Shift_1 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
  const __m256i bit1Shift_2 = _mm256_set1_epi32((1 << 23) + (1 << 16) + (1 << 9) + (1 << 2));
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    for (int i = 0; i < Unroll; i++) {
      memcpy(tmp + i * NTILE, zptr, NTILE * sizeof(int8_t));
    }
    for (int i = 0; i < UnpackLoop; i++) {
      v_zp_y[i] = _mm256_loadu_si256((const __m256i*)(tmp + i * 32));
      v_zp_y[i] = _mm256_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    int k_remain_unrll = utils::padto_le(k_remain, Unroll);
    int ib = 0;
    for (; ib < k_remain_unrll; ib += Unroll) {
      auto b4ptr = srcptr + (ir + ib) * NTILE / 2;
      auto b1ptr = bit1ptr + (ir + ib) * NTILE / 8;
      auto b2ptr = bit2ptr + (ir + ib) * NTILE / 4;
      for (int i = 0; i < UnpackLoop; i++) {
        auto v_s8_y = unpack_4bits(b4ptr + i * 16, vmask);
        auto vb1 = unpack_1bits(b1ptr + i * 4, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
        auto vb2 = unpack_2bits(b2ptr + i * 8, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        vb1 = _mm256_slli_epi32(vb1, 4);
        vb2 = _mm256_slli_epi32(vb2, 4);
        v_s8_y = _mm256_or_si256(v_s8_y, vb1);
        v_s8_y = _mm256_or_si256(v_s8_y, vb2);
        v_s8_y = _mm256_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(dstptr + i * 32 + (ir + ib) * NTILE), v_s8_y);
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
      for (int i = 0; i < UnpackLoop; i++) {
        auto v_s8_y = unpack_4bits((utils::bit4x2*)(tmpb4ptr + i * 16), vmask);
        auto vb1 = unpack_1bits((utils::bit1x8*)(tmpb1ptr + i * 4), bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
        auto vb2 = unpack_2bits((utils::bit2x4*)(tmpb2ptr + i * 8), vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        vb1 = _mm256_slli_epi32(vb1, 4);
        vb2 = _mm256_slli_epi32(vb2, 4);
        v_s8_y = _mm256_or_si256(v_s8_y, vb1);
        v_s8_y = _mm256_or_si256(v_s8_y, vb2);
        v_s8_y = _mm256_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(tmpout + i * 32), v_s8_y);
      }
      memcpy(dstptr + (ir + ib) * NTILE, tmpout, k_tail * NTILE);
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE decompress_s7_s8(utils::bit4x2* bit4ptr, utils::bit2x4* bit2ptr, utils::bit1x8* bit1ptr,
                                         int8_t* dstptr, size_t unpack_elt, int8_t* tmp, size_t tmpsize) {
  int constexpr VBits = 256;
  int constexpr VElt = VBits / 8;
  size_t i = 0;
  int constexpr FullRange = 1 << (7 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm256_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm256_set1_epi8(FullRange);

  uint32_t mask0 = 0x03030303;
  auto vmask0 = _mm256_set1_epi32(*(int32_t*)&mask0);
  auto vshift_y = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm256_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);

  const __m256i highMask = _mm256_set1_epi8(0x04);
  const __m256i bit1Mask = _mm256_set1_epi32(0x0F);
  const __m256i bit1Shift_1 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
  const __m256i bit1Shift_2 = _mm256_set1_epi32((1 << 23) + (1 << 16) + (1 << 9) + (1 << 2));
  size_t elt_pad = utils::padto_le(unpack_elt, VElt);
  for (; i < elt_pad; i += VElt) {
    auto vout = unpack_4bits(bit4ptr + i / 2, vmask);
    auto vb1 = unpack_1bits(bit1ptr + i / 8, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
    auto vb2 = unpack_2bits(bit2ptr + i / 4, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
    vb1 = _mm256_slli_epi32(vb1, 4);
    vb2 = _mm256_slli_epi32(vb2, 4);
    vout = _mm256_or_si256(vout, vb1);
    vout = _mm256_or_si256(vout, vb2);
    vout = _mm256_sub_epi8(vout, vbias);
    _mm256_storeu_si256((__m256i*)(dstptr + i), vout);
  }
  if (elt_pad < unpack_elt) {
    if (unpack_elt >= 32) {
      i = unpack_elt - 32;
      auto vout = unpack_4bits(bit4ptr + i / 2, vmask);
      auto vb1 = unpack_1bits(bit1ptr + i / 8, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
      auto vb2 = unpack_2bits(bit2ptr + i / 4, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
      vb1 = _mm256_slli_epi32(vb1, 4);
      vb2 = _mm256_slli_epi32(vb2, 4);
      vout = _mm256_or_si256(vout, vb1);
      vout = _mm256_or_si256(vout, vb2);
      vout = _mm256_sub_epi8(vout, vbias);
      _mm256_storeu_si256((__m256i*)(dstptr + i), vout);
    } else {
      ref::decompress_s7_s8(bit4ptr + i / 2, bit2ptr + i / 4, bit1ptr + i / 8, dstptr + i, unpack_elt - i, tmp,
                            tmpsize);
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

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s6_s8_pack4_row(utils::bit4x2* srcptr, utils::bit2x4* bit2ptr, int8_t* zpptr,
                                                          int8_t* dstptr, int blocksize, int ldzp, int n_offset,
                                                          int k_offset, int row, int8_t* tmp, size_t tmpsize) {
  int constexpr NReg = NTILE / 8;
  static_assert((NTILE % 4) == 0);
  int constexpr PackRow = 4;
  __m256i v_zp_y[NReg];
  const auto vindex = _mm256_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0);
  int constexpr FullRange = 1 << (6 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm256_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm256_set1_epi8(FullRange);

  uint32_t mask0 = 0x03030303;
  auto vmask0 = _mm256_set1_epi32(*(int32_t*)&mask0);
  auto vshift_y = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm256_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = load_zp_epi8_broadcast_epi32(zptr + i * 8, vindex);
      v_zp_y[i] = _mm256_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    for (int ib = 0; ib < k_remain; ib += PackRow) {
      auto b4ptr = srcptr + (ir + ib) * NTILE / 2;
      auto b2ptr = bit2ptr + (ir + ib) * NTILE / 4;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_4bits(b4ptr + i * 16, vmask);
        auto vb1 = unpack_2bits(b2ptr + i * 8, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        vb1 = _mm256_slli_epi32(vb1, 4);
        v_s8_y = _mm256_or_si256(v_s8_y, vb1);
        v_s8_y = _mm256_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(dstptr + i * 32 + (ir + ib) * NTILE), v_s8_y);
      }
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s6_s8_pack2_row(utils::bit4x2* srcptr, utils::bit2x4* bit2ptr, int8_t* zpptr,
                                                          int8_t* dstptr, int blocksize, int ldzp, int n_offset,
                                                          int k_offset, int row, int8_t* tmp, size_t tmpsize) {
  int constexpr NReg = NTILE / 8;
  static_assert((NTILE % 8) == 0);
  int constexpr PackRow = 2;
  int constexpr Unroll = 2;
  __m256i v_zp_y[NReg];
  const auto vindex = _mm256_set_epi8(14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0, 14, 14, 12, 12, 10, 10, 8,
                                      8, 6, 6, 4, 4, 2, 2, 0, 0);
  int constexpr FullRange = 1 << (6 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm256_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm256_set1_epi8(FullRange);

  uint32_t mask0 = 0x03030303;
  auto vmask0 = _mm256_set1_epi32(*(int32_t*)&mask0);
  auto vshift_y = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm256_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);

  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    memcpy(tmp, zptr, NTILE * sizeof(int8_t));
    memcpy(tmp + NTILE, zptr, NTILE * sizeof(int8_t));
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = load_zp_epi8_broadcast_epi16_v16(tmp + i * 16, vindex);
      v_zp_y[i] = _mm256_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    int k_remain_unrll = utils::padto_le(k_remain, PackRow * Unroll);
    int ib = 0;
    for (; ib < k_remain_unrll; ib += PackRow * Unroll) {
      auto b4ptr = srcptr + (ir + ib) * NTILE / 2;
      auto b2ptr = bit2ptr + (ir + ib) * NTILE / 4;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_4bits(b4ptr + i * 16, vmask);
        auto vb1 = unpack_2bits(b2ptr + i * 8, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        vb1 = _mm256_slli_epi32(vb1, 4);
        v_s8_y = _mm256_or_si256(v_s8_y, vb1);
        v_s8_y = _mm256_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(dstptr + i * 32 + (ir + ib) * NTILE), v_s8_y);
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
        auto v_s8_y = unpack_4bits((utils::bit4x2*)(tmpb4ptr + i * 16), vmask);
        auto vb1 = unpack_2bits((utils::bit2x4*)(tmpb2ptr + i * 8), vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        vb1 = _mm256_slli_epi32(vb1, 4);
        v_s8_y = _mm256_or_si256(v_s8_y, vb1);
        v_s8_y = _mm256_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(tmpout + i * 32), v_s8_y);
      }
      memcpy(dstptr + (ir + ib) * NTILE, tmpout, k_tail * NTILE);
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s6_s8_pack1_row(utils::bit4x2* srcptr, utils::bit2x4* bit2ptr, int8_t* zpptr,
                                                          int8_t* dstptr, int blocksize, int ldzp, int n_offset,
                                                          int k_offset, int row, int8_t* tmp, size_t tmpsize) {
  int constexpr NReg = NTILE / 8;
  static_assert((NTILE % 8) == 0);
  int constexpr PackRow = 1;
  int constexpr Unroll = 4;
  int constexpr UnpackLoop = Unroll * NTILE / 32;
  int constexpr FullRange = 1 << (6 - 1);
  __m256i v_zp_y[UnpackLoop];
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm256_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm256_set1_epi8(FullRange);

  uint32_t mask0 = 0x03030303;
  auto vmask0 = _mm256_set1_epi32(*(int32_t*)&mask0);
  auto vshift_y = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm256_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    for (int i = 0; i < Unroll; i++) {
      memcpy(tmp + i * NTILE, zptr, NTILE * sizeof(int8_t));
    }
    for (int i = 0; i < UnpackLoop; i++) {
      v_zp_y[i] = _mm256_loadu_si256((const __m256i*)(tmp + i * 32));
      v_zp_y[i] = _mm256_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    int k_remain_unrll = utils::padto_le(k_remain, Unroll);
    int ib = 0;
    for (; ib < k_remain_unrll; ib += Unroll) {
      auto b4ptr = srcptr + (ir + ib) * NTILE / 2;
      auto b2ptr = bit2ptr + (ir + ib) * NTILE / 4;
      for (int i = 0; i < UnpackLoop; i++) {
        auto v_s8_y = unpack_4bits(b4ptr + i * 16, vmask);
        auto vb1 = unpack_2bits(b2ptr + i * 8, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        vb1 = _mm256_slli_epi32(vb1, 4);
        v_s8_y = _mm256_or_si256(v_s8_y, vb1);
        v_s8_y = _mm256_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(dstptr + i * 32 + (ir + ib) * NTILE), v_s8_y);
      }
    }

    int k_tail = k_remain - k_remain_unrll;
    if (k_tail > 0) {
      auto tmpb4ptr = tmp;
      memcpy(tmpb4ptr, srcptr + (ir + ib) * NTILE / 2, k_tail * NTILE / 2);
      auto tmpb2ptr = tmp + Unroll * NTILE / 2;
      memcpy(tmpb2ptr, bit2ptr + (ir + ib) * NTILE / 4, k_tail * NTILE / 4);
      auto tmpout = tmp + Unroll * NTILE;
      for (int i = 0; i < UnpackLoop; i++) {
        auto v_s8_y = unpack_4bits((utils::bit4x2*)(tmpb4ptr + i * 16), vmask);
        auto vb1 = unpack_2bits((utils::bit2x4*)(tmpb2ptr + i * 8), vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        vb1 = _mm256_slli_epi32(vb1, 4);
        v_s8_y = _mm256_or_si256(v_s8_y, vb1);
        v_s8_y = _mm256_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm256_storeu_si256((__m256i*)(tmpout + i * 32), v_s8_y);
      }
      memcpy(dstptr + (ir + ib) * NTILE, tmpout, k_tail * NTILE);
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE decompress_s6_s8(utils::bit4x2* bit4ptr, utils::bit2x4* bit2ptr, int8_t* dstptr,
                                         size_t unpack_elt, int8_t* tmp, size_t tmpsize) {
  int constexpr VBits = 256;
  int constexpr VElt = VBits / 8;
  size_t i = 0;
  int constexpr FullRange = 1 << (6 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm256_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm256_set1_epi8(FullRange);

  uint32_t mask0 = 0x03030303;
  auto vmask0 = _mm256_set1_epi32(*(int32_t*)&mask0);
  auto vshift_y = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm256_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);
  size_t elt_pad = utils::padto_le(unpack_elt, VElt);
  for (; i < elt_pad; i += VElt) {
    auto vout = unpack_4bits(bit4ptr + i / 2, vmask);
    auto vb1 = unpack_2bits(bit2ptr + i / 4, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
    vb1 = _mm256_slli_epi32(vb1, 4);
    vout = _mm256_or_si256(vout, vb1);
    vout = _mm256_sub_epi8(vout, vbias);
    _mm256_storeu_si256((__m256i*)(dstptr + i), vout);
  }
  if (elt_pad < unpack_elt) {
    if (unpack_elt >= 32) {
      i = unpack_elt - 32;
      auto vout = unpack_4bits(bit4ptr + i / 2, vmask);
      auto vb1 = unpack_2bits(bit2ptr + i / 4, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
      vb1 = _mm256_slli_epi32(vb1, 4);
      vout = _mm256_or_si256(vout, vb1);
      vout = _mm256_sub_epi8(vout, vbias);
      _mm256_storeu_si256((__m256i*)(dstptr + i), vout);
    } else {
      ref::decompress_s6_s8(bit4ptr + i / 2, bit2ptr + i / 4, dstptr + i, unpack_elt - i, tmp, tmpsize);
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

template <bool WITH_SCALE, typename _DST_T, int _PACK_ROW, typename _S_T>
inline BTLA_CODE decompress_kblock_f8_fp(utils::f8* srcptr, _DST_T* dstptr, int row, int col, int ld_src, int ld_dst,
                                         _S_T* scales, int k_offset, int kblock, int NPad, BTLA_DTYPE src_f8_type) {
  int align_col = col / 16 * 16;
  int col_tail = col - align_col;
  auto ebits = utils::bestla_dtype_get_f8_ebits(src_f8_type);
  auto mantissabit = 7 - ebits;
  auto sign_revert_and_mask = _mm256_set1_epi32(0x80000000);
  auto e_revert_and_mask = _mm256_set1_epi32(0x0000007f);
  auto e_revert_shift = _mm256_set1_epi32(1);
  e_revert_shift = _mm256_slli_epi32(e_revert_shift, ebits - 1);
  e_revert_shift = _mm256_sub_epi32(e_revert_shift, _mm256_set1_epi32(128));
  auto mantissa_revert_and_mask = _mm256_set1_epi32(0x007fffff);
  auto packrow2_permute_idx = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
  for (int i = 0; i < row; i++) {
    int kpos = (k_offset + i) / kblock;
    auto sptr = scales + kpos * NPad;
    int j = 0;
    auto quant = [&]() {
      auto sign_revert = _mm256_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(srcptr + i * ld_src + j)));
      auto e_revert = sign_revert;
      auto mantissa_revert = sign_revert;
      sign_revert = _mm256_slli_epi32(sign_revert, 24);
      sign_revert = _mm256_and_si256(sign_revert, sign_revert_and_mask);
      e_revert = _mm256_and_si256(e_revert, e_revert_and_mask);
      e_revert = _mm256_srli_epi32(e_revert, mantissabit);
      if constexpr (WITH_SCALE && std::is_same_v<_S_T, utils::f8>) {
        auto scale = _mm256_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(sptr + j / _PACK_ROW)));
        if constexpr (_PACK_ROW == 2) scale = _mm256_permutevar8x32_epi32(packrow2_permute_idx, scale);
        e_revert = _mm256_add_epi32(e_revert, scale);
      }
      e_revert = _mm256_sub_epi32(e_revert, e_revert_shift);
      e_revert = _mm256_slli_epi32(e_revert, 23);
      mantissa_revert = _mm256_slli_epi32(mantissa_revert, 23 - mantissabit);
      mantissa_revert = _mm256_and_si256(mantissa_revert, mantissa_revert_and_mask);
      auto fp_v = _mm256_or_ps(_mm256_castsi256_ps(sign_revert), _mm256_castsi256_ps(e_revert));
      fp_v = _mm256_or_ps(fp_v, _mm256_castsi256_ps(mantissa_revert));
      if constexpr (WITH_SCALE && std::is_same_v<_S_T, float>) {
        auto scale = _mm256_loadu_ps(sptr + j / _PACK_ROW);
        if constexpr (_PACK_ROW == 2) scale = _mm256_permutevar8x32_ps(scale, packrow2_permute_idx);
        fp_v = _mm256_mul_ps(fp_v, scale);
      }
      if constexpr (std::is_same_v<_DST_T, float>) {
        _mm256_storeu_ps(dstptr + i * ld_dst + j, fp_v);
      } else {
        assert(0);
      }
    };
    for (; j < align_col; j += 8) quant();
    for (; j < col; j++) {
      auto fp_v = ref::f8_to_fp32(srcptr[i * ld_src + j], src_f8_type);
      if constexpr (WITH_SCALE) {
        if constexpr (std::is_same_v<_S_T, utils::f8>) {
          dstptr[i * ld_dst + j] = sptr[j / _PACK_ROW].mul(fp_v);
        } else if constexpr (std::is_same_v<_S_T, float>) {
          dstptr[i * ld_dst + j] = fp_v * sptr[j / _PACK_ROW];
        }
      } else {
        dstptr[i * ld_dst + j] = fp_v;
      }
    }
  }
  return BTLA_CODE::Success;
}

template <typename SCA_T>
static inline BTLA_CODE accum_alphaN_f32_f32(const SCA_T* alpha, const float* srcptr, const int srcstep, float* dstptr,
                                             const int dststep, const int M, const int N) {
  int constexpr Vlen = 8;
  auto vN = utils::padto_le(N, Vlen);
  int j = 0;
  for (; j < vN; j += Vlen) {
    __m256 valpha;
    if constexpr (std::is_same_v<SCA_T, float>) {
      valpha = _mm256_loadu_ps(alpha + j);
    } else if constexpr (std::is_same_v<SCA_T, utils::bf16>) {
      auto tmp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(alpha + j));
      valpha = ymm_cvt_bf16_fp32(tmp);
    } else if constexpr (std::is_same_v<SCA_T, utils::f8>) {
      auto ebit = _mm256_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(alpha + j)));
      ebit = _mm256_add_epi32(_mm256_set1_epi32(127), ebit);
      valpha = _mm256_castsi256_ps(_mm256_slli_epi32(ebit, 23));
    }
    for (size_t i = 0; i < M; i++) {
      auto vsrc = _mm256_loadu_ps(srcptr + i * srcstep + j);
      auto vsrc1 = _mm256_loadu_ps(dstptr + i * dststep + j);
      auto vdst = _mm256_fmadd_ps(valpha, vsrc, vsrc1);
      _mm256_storeu_ps(dstptr + i * dststep + j, vdst);
    }
  }
  for (; j < N; j += 1) {
    for (size_t i = 0; i < M; i++) {
      if constexpr (!std::is_same_v<SCA_T, utils::f8>) {
        dstptr[i * dststep + j] += alpha[j] * srcptr[i * srcstep + j];
      } else {
        dstptr[i * dststep + j] += alpha[j].mul(srcptr[i * srcstep + j]);
      }
    }
  }
  return BTLA_CODE::Success;
}

template <int N, typename _DST_T, BTLA_DTYPE F4_T, bool MULS_T>
static inline void dequant_f4_N(_DST_T* dstptr, int8_t* srcptr, __m256* vscales, __m256 vLutL, __m256 vLutH) {
  static_assert(N % 8 == 0);
  int constexpr VLoop = N / 8;
  auto v7 = _mm256_set1_epi32(7);
  auto v8 = _mm256_set1_epi32(8);
  for (int iv = 0; iv < VLoop; iv++) {
    auto idx = _mm_loadl_epi64(reinterpret_cast<__m128i*>(srcptr + iv * 8));
    auto pad_idx = _mm256_cvtepu8_epi32(idx);
    auto mskgt8 = _mm256_cmpgt_epi32(pad_idx, v7);
    auto fp32_dq_v0 = _mm256_permutevar8x32_ps(vLutL, pad_idx);
    pad_idx = _mm256_sub_epi32(pad_idx, v8);
    auto fp32_dq_v1 = _mm256_permutevar8x32_ps(vLutH, pad_idx);
    auto fp32_dq_v = _mm256_blendv_ps(fp32_dq_v0, fp32_dq_v1, _mm256_castsi256_ps(mskgt8));
    if constexpr (MULS_T) {
      fp32_dq_v = _mm256_mul_ps(fp32_dq_v, vscales[iv]);
    }
    if constexpr (std::is_same_v<_DST_T, float>) {
      _mm256_storeu_ps(dstptr + iv * 8, fp32_dq_v);
    } else if constexpr (std::is_same_v<_DST_T, utils::bf16>) {
      auto bf16v = ymm_cvt_fp32_bf16(fp32_dq_v);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(dstptr + iv * 8), bf16v);
    }
  }
}

template <int N, BTLA_DTYPE QT_T>
static inline void convert_s4_s8_N_avx2(int8_t* dstptr, int8_t* srcptr, __m256i mask) {
  static_assert(N % 2 == 0);
  static_assert(N <= 64);
  const auto vbias = _mm256_set1_epi8(8);
  if constexpr (N == 32) {
    auto dst0 = unpack_4bits(srcptr, mask);
    if constexpr (QT_T == BTLA_DTYPE::S4_CLIP) {
      dst0 = _mm256_sub_epi8(dst0, vbias);
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dstptr), dst0);
  } else if constexpr (N > 32) {
    auto dst0 = unpack_4bits(srcptr, mask);
    if constexpr (QT_T == BTLA_DTYPE::S4_CLIP) {
      dst0 = _mm256_sub_epi8(dst0, vbias);
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dstptr), dst0);
    int8_t temp[32];
    memcpy(temp, srcptr + 16, (N - 32) / 2);
    dst0 = unpack_4bits(temp, mask);
    if constexpr (QT_T == BTLA_DTYPE::S4_CLIP) {
      dst0 = _mm256_sub_epi8(dst0, vbias);
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(temp), dst0);
    memcpy(dstptr + 32, temp, (N - 32));
  } else {
    int8_t temp[32];
    memcpy(temp, srcptr, N / 2);
    auto dst0 = unpack_4bits(temp, mask);
    if constexpr (QT_T == BTLA_DTYPE::S4_CLIP) {
      dst0 = _mm256_sub_epi8(dst0, vbias);
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(temp), dst0);
    memcpy(dstptr, temp, N);
  }
}

template <BTLA_DTYPE F4_T, typename DST_T>
inline BTLA_CODE decompress_kblock_f4_fp_noscale(utils::f4x2* srcptr, DST_T* dstptr, int row, int col, int ld_src,
                                                 int ld_dst, int8_t* tmp, size_t tmpsize) {
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm256_set1_epi32(*reinterpret_cast<int*>(&mask));
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
  auto vLutL = _mm256_loadu_ps(LUT);
  auto vLutH = _mm256_loadu_ps(LUT + 8);
  if (col == ld_src) {
    size_t elesize = static_cast<size_t>(row) * col;
    size_t velt = utils::padto_le(elesize, 32);
    size_t i = 0;
    assert(tmpsize >= 32);
    for (; i < velt; i += 32) {
      convert_s4_s8_N_avx2<32, F4_T>(tmp, reinterpret_cast<int8_t*>(srcptr + i / 2), vmask);
      dequant_f4_N<32, DST_T, F4_T, false>(dstptr + i, tmp, nullptr, vLutL, vLutH);
    }
    for (; i < elesize; i += 2) {
      auto tmp = srcptr[i / 2];
      dstptr[i + 0] = static_cast<DST_T>(ref::f4_unpack<F4_T>(tmp.x));
      dstptr[i + 1] = static_cast<DST_T>(ref::f4_unpack<F4_T>(tmp.y));
    }
    return BTLA_CODE::Success;
  }
  return BTLA_CODE::Success;
}

template <BTLA_DTYPE QT_T, bool _IS_SYM, int _NCOL, typename _ST, typename _DST_T>
static inline BTLA_CODE decompress_kblock_bit4_packrow1(utils::bit4x2* srcptr, _DST_T* dstptr, int row, int col,
                                                        int ld_src, int ld_dst, _ST* scales, int8_t* zero_points,
                                                        int k_offset, int kblock, int NPad, int8_t* tmpbuf,
                                                        size_t tmpsize) {
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm256_set1_epi32(*reinterpret_cast<int*>(&mask));
  float* LUT = nullptr;
  if constexpr (QT_T == BTLA_DTYPE::F4_BNB) {
    LUT = fp4_bnb_dequant_fp32_LUT;
  } else if constexpr (QT_T == BTLA_DTYPE::F4_NF4) {
    LUT = nf4_dequant_fp32_LUT;
  } else if constexpr (QT_T == BTLA_DTYPE::F4_E2M1) {
    LUT = fp4_e2m1_dequant_fp32_LUT;
  }
  __m256 vLutL, vLutH;
  if (LUT) {
    vLutL = _mm256_loadu_ps(LUT);
    vLutH = _mm256_loadu_ps(LUT + 8);
  }
  int constexpr NReg = _NCOL / 8;
  assert(col == _NCOL);
  assert(ld_src == _NCOL);
  assert(ld_dst == _NCOL);
  __m256 vscales[NReg];
  __m256i vzps[NReg];
  int constexpr UnrollRow = 4;
  assert(kblock % UnrollRow == 0);
  int constexpr NTile = 32;
  int constexpr Loop32 = _NCOL * UnrollRow / NTile;
  assert(tmpsize >= (_NCOL * UnrollRow));
  int row0 = kblock - k_offset % kblock;
  row0 = row0 == kblock ? 0 : row0;
  row0 = row0 > row ? row : row0;
  int row1 = row - row0;
  int irow = 0;
  auto dequantize = [&](_DST_T* dstptr, int8_t* srcptr, __m256* vscales, __m256i* vzps = nullptr) {
    if constexpr (QT_T == BTLA_DTYPE::S4_CLIP) {
      dequant_s8_N_avx2<_NCOL, _IS_SYM>(dstptr, srcptr, vscales, vzps);
    } else {
      dequant_f4_N<_NCOL, _DST_T, QT_T, true>(dstptr, srcptr, vscales, vLutL, vLutH);
    }
  };
  if (row0) {
    int rowpad4 = utils::padto_le(row0, UnrollRow);
    for (int iv = 0; iv < NReg; iv++) {
      vscales[iv] = _mm256_loadu_ps(scales + (k_offset + irow) / kblock * NPad + iv * 8);
      if constexpr (!_IS_SYM) {
        auto tmp =
            _mm_loadl_epi64(reinterpret_cast<__m128i*>(zero_points + (k_offset + irow) / kblock * NPad + iv * 8));
        vzps[iv] = _mm256_cvtepi8_epi32(tmp);
      }
    }
    for (; irow < rowpad4; irow += UnrollRow) {
      for (int iter16 = 0; iter16 < Loop32; iter16++)
        convert_s4_s8_N_avx2<NTile, QT_T>(
            tmpbuf + iter16 * NTile, reinterpret_cast<int8_t*>(srcptr + irow * ld_src / 2 + NTile / 2 * iter16), vmask);
      for (int iterr = 0; iterr < UnrollRow; iterr++)
        dequantize(dstptr + (irow + iterr) * ld_dst, tmpbuf + iterr * _NCOL, vscales, vzps);
    }
    for (; irow < row0; irow++) {
      convert_s4_s8_N_avx2<_NCOL, QT_T>(tmpbuf, reinterpret_cast<int8_t*>(srcptr + irow * ld_src / 2), vmask);

      dequantize(dstptr + irow * ld_dst, tmpbuf, vscales, vzps);
    }
  }

  int row1_blk = utils::padto_le(row1, kblock) + row0;
  for (; irow < row1_blk; irow += kblock) {
    for (int iv = 0; iv < NReg; iv++) {
      vscales[iv] = _mm256_loadu_ps(scales + (k_offset + irow) / kblock * NPad + iv * 8);
      if constexpr (!_IS_SYM) {
        auto tmp =
            _mm_loadl_epi64(reinterpret_cast<__m128i*>(zero_points + (k_offset + irow) / kblock * NPad + iv * 8));
        vzps[iv] = _mm256_cvtepi8_epi32(tmp);
      }
    }
    for (int irr = 0; irr < kblock; irr += UnrollRow) {
      for (int iter16 = 0; iter16 < Loop32; iter16++)
        convert_s4_s8_N_avx2<NTile, QT_T>(
            tmpbuf + iter16 * NTile, reinterpret_cast<int8_t*>(srcptr + (irow + irr) * ld_src / 2 + NTile / 2 * iter16),
            vmask);
      for (int iterr = 0; iterr < UnrollRow; iterr++)
        dequantize(dstptr + (irow + irr + iterr) * ld_src, tmpbuf + iterr * _NCOL, vscales, vzps);
    }
  }
  if (irow < row) {
    for (int iv = 0; iv < NReg; iv++) {
      vscales[iv] = _mm256_loadu_ps(scales + (k_offset + irow) / kblock * NPad + iv * 8);
      if constexpr (!_IS_SYM) {
        auto tmp =
            _mm_loadl_epi64(reinterpret_cast<__m128i*>(zero_points + (k_offset + irow) / kblock * NPad + iv * 8));
        vzps[iv] = _mm256_cvtepi8_epi32(tmp);
      }
    }
    auto rowre = row - irow;
    int rowpad4 = utils::padto_le(rowre, UnrollRow) + irow;
    for (; irow < rowpad4; irow += UnrollRow) {
      for (int iter16 = 0; iter16 < Loop32; iter16++)
        convert_s4_s8_N_avx2<NTile, QT_T>(
            tmpbuf + iter16 * NTile, reinterpret_cast<int8_t*>(srcptr + irow * ld_src / 2 + NTile / 2 * iter16), vmask);
      for (int iterr = 0; iterr < UnrollRow; iterr++)
        dequantize(dstptr + (irow + iterr) * ld_dst, tmpbuf + iterr * _NCOL, vscales, vzps);
    }
    for (; irow < row; irow++) {
      convert_s4_s8_N_avx2<_NCOL, QT_T>(tmpbuf, reinterpret_cast<int8_t*>(srcptr + irow * ld_src / 2), vmask);
      dequantize(dstptr + irow * ld_dst, tmpbuf, vscales, vzps);
    }
  }
  return BTLA_CODE::Success;
}

template <BTLA_DTYPE S4_T, bool _IS_SYM, typename _ST, typename _DST_T>
static inline BTLA_CODE decompress_kblock_bit4_packrow2(utils::bit4x2* srcptr, _DST_T* dstptr, int row, int col,
                                                        int ld_src, int ld_dst, _ST* scales, int8_t* zero_points,
                                                        int k_offset, int kblock, int NPad, int8_t* tmp,
                                                        size_t tmpsize) {
  return BTLA_CODE::NotSupport;
}

template <int PackRow, int NTILE, typename DST_T>
inline BTLA_CODE decompress_kblock_s8_fp_row(int8_t* srcptr, DST_T* dstptr, int row, void* scales_, BTLA_DTYPE sdtype,
                                             int8_t* zero_points, int k_offset, int n_offset, int blocksize, int ldzp,
                                             int8_t* tmp, size_t tmpsize) {
  int constexpr NReg = NTILE / 8;
  const auto DstSize = row * NTILE * sizeof(DST_T);
  const auto S8Size = row * NTILE * sizeof(int8_t);
  if (zero_points == nullptr) {
    for (int ir = 0; ir < row; ir += blocksize) {
      int k_remain = utils::remainsize(ir, row, blocksize);
      int ele_off = (k_offset + ir) / blocksize * ldzp + n_offset;
      if constexpr (PackRow == 1) {
        __m256 vscale_y[NReg];
        if (sdtype == BTLA_DTYPE::F32) {
          auto sptr = (float*)scales_ + ele_off;
          for (int i = 0; i < NReg; i++) vscale_y[i] = _mm256_loadu_ps(sptr + i * 8);
        } else if (sdtype == BTLA_DTYPE::BF16) {
          auto sptr = (utils::bf16*)scales_ + ele_off;
          for (int i = 0; i < NReg; i++) vscale_y[i] = load_bf16_fp32(sptr + i * 8);
        } else if (sdtype == BTLA_DTYPE::F16) {
          auto sptr = (utils::fp16*)scales_ + ele_off;
          for (int i = 0; i < NReg; i++) vscale_y[i] = load_fp16_fp32(sptr + i * 8);
        }
        for (int ib = 0; ib < k_remain; ib += PackRow) {
          auto b8ptr = srcptr + (ir + ib) * NTILE;
          for (int i = 0; i < NReg; i++) {
            auto vdeq_y = dequant_s8_fp(b8ptr + i * 8, vscale_y[i]);
            store_fp32_T(vdeq_y, dstptr + (ir + ib) * NTILE + i * 8);
          }
        }
      } else if constexpr (PackRow == 4) {
        const auto vshuf_index_y = _mm256_set_epi8(15, 14, 13, 12, 15, 14, 13, 12, 11, 10, 9, 8, 11, 10, 9, 8, 7, 6, 5,
                                                   4, 7, 6, 5, 4, 3, 2, 1, 0, 3, 2, 1, 0);
        __m256 vscale_y[PackRow * NReg];
        for (int i = 0; i < NReg; i++) {
          __m256 vraw;
          if (sdtype == BTLA_DTYPE::F32) {
            auto sptr = (float*)scales_ + ele_off;
            vraw = _mm256_loadu_ps(sptr + i * 8);
          } else if (sdtype == BTLA_DTYPE::BF16) {
            auto sptr = (utils::bf16*)scales_ + ele_off;
            vraw = load_bf16_fp32(sptr + i * 8);
          } else if (sdtype == BTLA_DTYPE::F16) {
            auto sptr = (utils::fp16*)scales_ + ele_off;
            vraw = load_fp16_fp32(sptr + i * 8);
          } else {
            assert(0);
          }
          auto vcast_y = broadcast_ps_1_2<true>(vraw, vshuf_index_y);
          vscale_y[i * PackRow + 0] = broadcast_ps_1_2<true>(vcast_y, vshuf_index_y);
          vscale_y[i * PackRow + 1] = broadcast_ps_1_2<false>(vcast_y, vshuf_index_y);
          vcast_y = broadcast_ps_1_2<false>(vraw, vshuf_index_y);
          vscale_y[i * PackRow + 2] = broadcast_ps_1_2<true>(vcast_y, vshuf_index_y);
          vscale_y[i * PackRow + 3] = broadcast_ps_1_2<false>(vcast_y, vshuf_index_y);
        }
        for (int ib = 0; ib < k_remain; ib += PackRow) {
          auto b8ptr = srcptr + (ir + ib) * NTILE;
          for (int i = 0; i < NReg; i++) {
            for (int ip = 0; ip < PackRow; ip++) {
              auto vdeq_y = dequant_s8_fp(b8ptr + i * 8 * PackRow + ip * 8, vscale_y[i * PackRow + ip]);
              store_fp32_T(vdeq_y, dstptr + (ir + ib) * NTILE + i * 8 * PackRow + ip * 8);
            }
          }
        }
      } else if constexpr (PackRow == 2) {
        const auto vshuf_index_y = _mm256_set_epi8(15, 14, 13, 12, 15, 14, 13, 12, 11, 10, 9, 8, 11, 10, 9, 8, 7, 6, 5,
                                                   4, 7, 6, 5, 4, 3, 2, 1, 0, 3, 2, 1, 0);
        __m256 vscale_y[PackRow * NReg];
        for (int i = 0; i < NReg; i++) {
          __m256 vraw;
          if (sdtype == BTLA_DTYPE::F32) {
            auto sptr = (float*)scales_ + ele_off;
            vraw = _mm256_loadu_ps(sptr + i * 8);
          } else if (sdtype == BTLA_DTYPE::BF16) {
            auto sptr = (utils::bf16*)scales_ + ele_off;
            vraw = load_bf16_fp32(sptr + i * 8);
          } else if (sdtype == BTLA_DTYPE::F16) {
            auto sptr = (utils::fp16*)scales_ + ele_off;
            vraw = load_fp16_fp32(sptr + i * 8);
          }
          vscale_y[i * PackRow + 0] = broadcast_ps_1_2<true>(vraw, vshuf_index_y);
          vscale_y[i * PackRow + 1] = broadcast_ps_1_2<false>(vraw, vshuf_index_y);
        }
        for (int ib = 0; ib < k_remain; ib += PackRow) {
          auto b8ptr = srcptr + (ir + ib) * NTILE;
          for (int i = 0; i < NReg; i++) {
            for (int ip = 0; ip < PackRow; ip++) {
              auto vdeq_y = dequant_s8_fp(b8ptr + i * 8 * PackRow + ip * 8, vscale_y[i * PackRow + ip]);
              store_fp32_T(vdeq_y, dstptr + (ir + ib) * NTILE + i * 8 * PackRow + ip * 8);
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
        __m256 vscale_y[NReg];
        if (sdtype == BTLA_DTYPE::F32) {
          auto sptr = (float*)scales_ + ele_off;
          for (int i = 0; i < NReg; i++) vscale_y[i] = _mm256_loadu_ps(sptr + i * 8);
        } else if (sdtype == BTLA_DTYPE::BF16) {
          auto sptr = (utils::bf16*)scales_ + ele_off;
          for (int i = 0; i < NReg; i++) vscale_y[i] = load_bf16_fp32(sptr + i * 8);
        } else if (sdtype == BTLA_DTYPE::F16) {
          auto sptr = (utils::fp16*)scales_ + ele_off;
          for (int i = 0; i < NReg; i++) vscale_y[i] = load_fp16_fp32(sptr + i * 8);
        }
        __m256i vzp_y[NReg];
        for (int i = 0; i < NReg; i++) vzp_y[i] = load_s8_s32(zero_points + ele_off + i * 8);
        for (int ib = 0; ib < k_remain; ib += PackRow) {
          auto b8ptr = srcptr + (ir + ib) * NTILE;
          for (int i = 0; i < NReg; i++) {
            auto vdeq_y = dequant_s8_fp<true>(b8ptr + i * 8, vscale_y[i], vzp_y[i]);
            store_fp32_T(vdeq_y, dstptr + (ir + ib) * NTILE + i * 8);
          }
        }
      } else if constexpr (PackRow == 4) {
        const auto vshuf_index_y = _mm256_set_epi8(15, 14, 13, 12, 15, 14, 13, 12, 11, 10, 9, 8, 11, 10, 9, 8, 7, 6, 5,
                                                   4, 7, 6, 5, 4, 3, 2, 1, 0, 3, 2, 1, 0);
        __m256 vscale_y[PackRow * NReg];
        __m256i vzp_y[PackRow * NReg];
        for (int i = 0; i < NReg; i++) {
          __m256 vraw;
          if (sdtype == BTLA_DTYPE::F32) {
            auto sptr = (float*)scales_ + ele_off;
            vraw = _mm256_loadu_ps(sptr + i * 8);
          } else if (sdtype == BTLA_DTYPE::BF16) {
            auto sptr = (utils::bf16*)scales_ + ele_off;
            vraw = load_bf16_fp32(sptr + i * 8);
          } else if (sdtype == BTLA_DTYPE::F16) {
            auto sptr = (utils::fp16*)scales_ + ele_off;
            vraw = load_fp16_fp32(sptr + i * 8);
          } else {
            assert(0);
          }
          auto vcast_y = broadcast_ps_1_2<true>(vraw, vshuf_index_y);
          vscale_y[i * PackRow + 0] = broadcast_ps_1_2<true>(vcast_y, vshuf_index_y);
          vscale_y[i * PackRow + 1] = broadcast_ps_1_2<false>(vcast_y, vshuf_index_y);
          vcast_y = broadcast_ps_1_2<false>(vraw, vshuf_index_y);
          vscale_y[i * PackRow + 2] = broadcast_ps_1_2<true>(vcast_y, vshuf_index_y);
          vscale_y[i * PackRow + 3] = broadcast_ps_1_2<false>(vcast_y, vshuf_index_y);

          auto tmp = load_s8_s32(zero_points + ele_off + i * 8);
          auto vcasti_y = broadcast_epi32_1_2<true>(tmp, vshuf_index_y);
          vzp_y[i * PackRow + 0] = broadcast_epi32_1_2<true>(vcasti_y, vshuf_index_y);
          vzp_y[i * PackRow + 1] = broadcast_epi32_1_2<false>(vcasti_y, vshuf_index_y);
          vcasti_y = broadcast_epi32_1_2<false>(tmp, vshuf_index_y);
          vzp_y[i * PackRow + 2] = broadcast_epi32_1_2<true>(vcasti_y, vshuf_index_y);
          vzp_y[i * PackRow + 3] = broadcast_epi32_1_2<false>(vcasti_y, vshuf_index_y);
        }
        for (int ib = 0; ib < k_remain; ib += PackRow) {
          auto b8ptr = srcptr + (ir + ib) * NTILE;
          for (int i = 0; i < NReg; i++) {
            for (int ip = 0; ip < PackRow; ip++) {
              auto vdeq_y = dequant_s8_fp<true>(b8ptr + i * 8 * PackRow + ip * 8, vscale_y[i * PackRow + ip],
                                                vzp_y[i * PackRow + ip]);
              store_fp32_T(vdeq_y, dstptr + (ir + ib) * NTILE + i * 8 * PackRow + ip * 8);
            }
          }
        }
      } else if constexpr (PackRow == 2) {
        const auto vshuf_index_y = _mm256_set_epi8(15, 14, 13, 12, 15, 14, 13, 12, 11, 10, 9, 8, 11, 10, 9, 8, 7, 6, 5,
                                                   4, 7, 6, 5, 4, 3, 2, 1, 0, 3, 2, 1, 0);
        __m256 vscale_y[PackRow * NReg];
        __m256i vzp_y[PackRow * NReg];
        for (int i = 0; i < NReg; i++) {
          __m256 vraw;
          if (sdtype == BTLA_DTYPE::F32) {
            auto sptr = (float*)scales_ + ele_off;
            vraw = _mm256_loadu_ps(sptr + i * 8);
          } else if (sdtype == BTLA_DTYPE::BF16) {
            auto sptr = (utils::bf16*)scales_ + ele_off;
            vraw = load_bf16_fp32(sptr + i * 8);
          }
          vscale_y[i * PackRow + 0] = broadcast_ps_1_2<true>(vraw, vshuf_index_y);
          vscale_y[i * PackRow + 1] = broadcast_ps_1_2<false>(vraw, vshuf_index_y);
          auto tmp = load_s8_s32(zero_points + ele_off + i * 8);
          vzp_y[i * PackRow + 0] = broadcast_epi32_1_2<true>(tmp, vshuf_index_y);
          vzp_y[i * PackRow + 1] = broadcast_epi32_1_2<false>(tmp, vshuf_index_y);
        }
        for (int ib = 0; ib < k_remain; ib += PackRow) {
          auto b8ptr = srcptr + (ir + ib) * NTILE;
          for (int i = 0; i < NReg; i++) {
            for (int ip = 0; ip < PackRow; ip++) {
              auto vdeq_y = dequant_s8_fp<true>(b8ptr + i * 8 * PackRow + ip * 8, vscale_y[i * PackRow + ip],
                                                vzp_y[i * PackRow + ip]);
              store_fp32_T(vdeq_y, dstptr + (ir + ib) * NTILE + i * 8 * PackRow + ip * 8);
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

template <BTLA_DTYPE _F4_T, typename _DST_T, int _PACK_ROW, typename _ST>
static inline BTLA_CODE decompress_kblock_f4_fp(utils::f4x2* srcptr, _DST_T* dstptr, int row, int col, int ld_src,
                                                int ld_dst, _ST* scales, int k_offset, int kblock, int NPad,
                                                int8_t* tmp, size_t tmpsize) {
  if constexpr (_PACK_ROW == 1) {
    if (col == 24) {
      return decompress_kblock_bit4_packrow1<_F4_T, true, 24, _ST, _DST_T>(
          srcptr, dstptr, row, col, ld_src, ld_dst, scales, nullptr, k_offset, kblock, NPad, tmp, tmpsize);
    }
    if (col == 48) {
      return decompress_kblock_bit4_packrow1<_F4_T, true, 48, _ST, _DST_T>(
          srcptr, dstptr, row, col, ld_src, ld_dst, scales, nullptr, k_offset, kblock, NPad, tmp, tmpsize);
    }
  } else if constexpr (_PACK_ROW == 2) {
    return decompress_kblock_bit4_packrow2<_F4_T, true, _ST, _DST_T>(srcptr, dstptr, row, col, ld_src, ld_dst, scales,
                                                                     nullptr, k_offset, kblock, NPad, tmp, tmpsize);
  }
  assert(0);
  return BTLA_CODE::NotSupport;
}

enum class AVX2_REDUCE_TYPE { MAX, MIN, ADD };
#define AVX2_REDUCE_OP                                                  \
  if constexpr (TYPE == AVX2_REDUCE_TYPE::MAX) x = _mm256_max_ps(x, y); \
  if constexpr (TYPE == AVX2_REDUCE_TYPE::MIN) x = _mm256_min_ps(x, y); \
  if constexpr (TYPE == AVX2_REDUCE_TYPE::ADD) x = _mm256_add_ps(x, y);

template <AVX2_REDUCE_TYPE TYPE>
inline float avx2_reduce_ps(__m256 x) {
  __m256 y = _mm256_permute2f128_ps(x, x, 1);
  AVX2_REDUCE_OP
  y = _mm256_permute_ps(x, 0b01001110);
  AVX2_REDUCE_OP
  y = _mm256_permute_ps(x, 0b10110001);
  AVX2_REDUCE_OP
  return _mm256_cvtss_f32(x);
}

#define AVX2_REDUCE_OP_EPI32(dst, src)                                           \
  if constexpr (TYPE == AVX2_REDUCE_TYPE::MAX) dst = _mm256_max_epi32(dst, src); \
  if constexpr (TYPE == AVX2_REDUCE_TYPE::MIN) dst = _mm256_min_epi32(dst, src); \
  if constexpr (TYPE == AVX2_REDUCE_TYPE::ADD) dst = _mm256_add_epi32(dst, src);

#ifndef _mm256_cvtsi256_si32
#define _mm256_cvtsi256_si32(a) (_mm_cvtsi128_si32(_mm256_castsi256_si128(a)))
#endif

template <AVX2_REDUCE_TYPE TYPE>
inline int avx2_reduce_epi32(__m256i xd) {
  auto x = _mm256_castsi256_ps(xd);
  __m256 y = _mm256_permute2f128_ps(x, x, 1);
  auto yd = _mm256_castps_si256(y);
  AVX2_REDUCE_OP_EPI32(xd, yd);
  x = _mm256_castsi256_ps(xd);
  y = _mm256_permute_ps(x, 0b01001110);
  yd = _mm256_castps_si256(y);
  AVX2_REDUCE_OP_EPI32(xd, yd);
  x = _mm256_castsi256_ps(xd);
  y = _mm256_permute_ps(x, 0b10110001);
  yd = _mm256_castps_si256(y);
  AVX2_REDUCE_OP_EPI32(xd, yd);
  return _mm256_cvtsi256_si32(xd);
}

inline __m128i avx2_cvtepi32_epu8(__m256i x) {
  auto out_v = _mm_packus_epi32(_mm256_castsi256_si128(x), _mm256_extractf128_si256(x, 1));
  out_v = _mm_packus_epi16(out_v, out_v);
  return out_v;
}

template <typename SRC_T>
static inline BTLA_CODE quantize_fp_u8_colblock(int row, int col, const SRC_T* srcptr, int ld_src, uint8_t* dstptr,
                                                int ld_dst, float* scales, int ld_scale, uint8_t* zps, int blocksize,
                                                float* blkreduce) {
  int constexpr VLen = 8;
  auto vff = _mm256_set1_epi32(255);
  auto v0 = _mm256_set1_epi32(0);
  int constexpr Unroll = 2;
  int vblocksize_un = utils::padto_le(blocksize, VLen * Unroll);
  int vblocksize = utils::padto_le(blocksize, VLen);
  int colblk = utils::padto_le(col, blocksize);
  for (size_t i = 0; i < row; i++) {
    size_t j = 0;
    for (; j < colblk; j += blocksize) {
      __m256 vmaxval = _mm256_set1_ps(0.f);
      __m256 vminval = _mm256_set1_ps(0.f);
      size_t ij = 0;
      for (; ij < vblocksize_un; ij += VLen * Unroll) {
        for (size_t iu = 0; iu < Unroll; iu++) {
          __m256 vsrc = load_T_fp32(&srcptr[(j + ij) + i * ld_src + iu * VLen]);
          vmaxval = _mm256_max_ps(vmaxval, vsrc);
          vminval = _mm256_min_ps(vminval, vsrc);
        }
      }
      if (ij + VLen < vblocksize) {
        for (; ij < vblocksize; ij += VLen) {
          __m256 vsrc = load_T_fp32(&srcptr[(j + ij) + i * ld_src]);
          vmaxval = _mm256_max_ps(vmaxval, vsrc);
          vminval = _mm256_min_ps(vminval, vsrc);
        }
      }
      auto maxval = avx2_reduce_ps<AVX2_REDUCE_TYPE::MAX>(vmaxval);
      auto minval = avx2_reduce_ps<AVX2_REDUCE_TYPE::MIN>(vminval);
      if (ij < blocksize) {
        for (; ij < blocksize; ij++) {
          auto srcval = (float)srcptr[(j + ij) + i * ld_src];
          maxval = std::max(maxval, srcval);
          minval = std::min(minval, srcval);
        }
      }
      float scale = (maxval - minval) / 255;
      uint8_t zp = utils::cast<float, uint8_t>((0 - minval) / scale);
      scales[j / blocksize + i * ld_scale] = scale;
      zps[j / blocksize + i * ld_scale] = zp;
      int sum = 0;
      float rscale = 1.f / scale;
      auto vrscale = _mm256_set1_ps(rscale);
      auto vdzp = _mm256_set1_epi32(zp);
      ij = 0;
      if (blkreduce) {
        for (; ij < vblocksize; ij += VLen) {
          __m256 vsrc = load_T_fp32(&srcptr[(j + ij) + i * ld_src]);
          vsrc = _mm256_mul_ps(vsrc, vrscale);
          auto vdsrc = _mm256_cvtps_epi32(vsrc);
          sum += avx2_reduce_epi32<AVX2_REDUCE_TYPE::ADD>(vdsrc);
          vdsrc = _mm256_add_epi32(vdsrc, vdzp);
          vdsrc = _mm256_min_epi32(vdsrc, vff);
          vdsrc = _mm256_max_epi32(vdsrc, v0);
          auto vbsrc = avx2_cvtepi32_epu8(vdsrc);
          _mm_storel_epi64(reinterpret_cast<__m128i*>(&dstptr[(j + ij) + i * ld_dst]), vbsrc);
        }
      } else {
        for (; ij < vblocksize_un; ij += VLen * Unroll) {
          for (size_t iu = 0; iu < Unroll; iu++) {
            __m256 vsrc = load_T_fp32(&srcptr[(j + ij) + i * ld_src + iu * VLen]);
            vsrc = _mm256_mul_ps(vsrc, vrscale);
            auto vdsrc = _mm256_cvtps_epi32(vsrc);
            vdsrc = _mm256_add_epi32(vdsrc, vdzp);
            vdsrc = _mm256_min_epi32(vdsrc, vff);
            vdsrc = _mm256_max_epi32(vdsrc, v0);
            auto vbsrc = avx2_cvtepi32_epu8(vdsrc);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(&dstptr[(j + ij) + i * ld_dst + iu * VLen]), vbsrc);
          }
        }
        if (ij + VLen < vblocksize) {
          for (; ij < vblocksize; ij += VLen) {
            __m256 vsrc = load_T_fp32(&srcptr[(j + ij) + i * ld_src]);
            vsrc = _mm256_mul_ps(vsrc, vrscale);
            auto vdsrc = _mm256_cvtps_epi32(vsrc);
            vdsrc = _mm256_add_epi32(vdsrc, vdzp);
            vdsrc = _mm256_min_epi32(vdsrc, vff);
            vdsrc = _mm256_max_epi32(vdsrc, v0);
            auto vbsrc = avx2_cvtepi32_epu8(vdsrc);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(&dstptr[(j + ij) + i * ld_dst]), vbsrc);
          }
        }
      }
      for (; ij < blocksize; ij++) {
        auto srcval = (float)srcptr[(j + ij) + i * ld_src];
        srcval = srcval * rscale;
        auto srcint = int(roundf(srcval));
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
        maxval = std::max((float)srcptr[ij + i * ld_src], maxval);
        minval = std::min((float)srcptr[ij + i * ld_src], minval);
      }
      float scale = (maxval - minval) / 255;
      uint8_t zp = utils::cast<float, uint8_t>((0 - minval) / scale);
      float rscale = 1.f / scale;
      scales[j / blocksize + i * ld_scale] = scale;
      zps[j / blocksize + i * ld_scale] = zp;
      int sum = 0;
      for (size_t ij = j; ij < col; ij++) {
        auto srcint = utils::cast<float, int>(srcptr[ij + i * ld_src] * rscale);
        sum += srcint;
        srcint += zp;
        srcint = srcint <= 255 ? srcint : 255;
        srcint = srcint >= 0 ? srcint : 0;
        dstptr[ij + i * ld_dst] = utils::cast<int, uint8_t>(srcint);
      }
      if (blkreduce) {
        blkreduce[j / blocksize + i * ld_scale] = sum * scale;
      }
    }
  }
  return BTLA_CODE::Success;
}

template <typename SRC_T>
static inline BTLA_CODE col_block_reduce_sum(const SRC_T* srcptr, int ldsrc, int row, int col, int blocksize,
                                             float* reduce, int ldr) {
  int constexpr VLen = 8;
  auto vblock2_ = utils::padto_le(blocksize, VLen * 2);
  auto vblock_ = utils::padto_le(blocksize, VLen);
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j += blocksize) {
      auto tmp = 0.f;
      auto vsum = _mm256_set1_ps(0.f);
      int jj = 0;
      auto vblock2 = j + vblock2_ <= col ? vblock2_ : 0;
      auto vblock = j + vblock_ <= col ? vblock_ : 0;
      for (; jj < vblock2; jj += VLen * 2) {
        auto vtmp = _mm256_loadu_ps(srcptr + i * ldsrc + j + jj);
        auto vtmp1 = _mm256_loadu_ps(srcptr + i * ldsrc + j + jj + VLen);
        auto s0 = avx2_reduce_ps<AVX2_REDUCE_TYPE::ADD>(vtmp);
        auto s1 = avx2_reduce_ps<AVX2_REDUCE_TYPE::ADD>(vtmp1);
        tmp += s0;
        tmp += s1;
      }
      if (jj + VLen <= vblock) {
        for (; jj < vblock; jj += VLen) {
          auto vtmp = _mm256_loadu_ps(srcptr + i * ldsrc + j + jj);
          auto s0 = avx2_reduce_ps<AVX2_REDUCE_TYPE::ADD>(vtmp);
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

template <typename T>
static inline BTLA_CODE cvt_fp32_T_2D(const float* src_ptr, T* dst_ptr, int row, int col, int src_step, int dst_step,
                                      bool zeropadding) {
  const int npadding = (dst_step - col) * sizeof(T);
  constexpr int simd_proc_elt = 8;
  float tmpbuf[simd_proc_elt];
  auto col_body = col / simd_proc_elt * simd_proc_elt;
  auto col_tail = col % simd_proc_elt;
  for (int i = 0; i < row; i++) {
    const auto src = src_ptr + i * src_step;
    const auto dst = dst_ptr + i * dst_step;
    int j = 0;
    for (; j < col_body; j += simd_proc_elt) {
      store_fp32_T(_mm256_loadu_ps(src + j), dst + j);
    }
    if (col_tail > 0) {
      memcpy(tmpbuf, src + j, col_tail * sizeof(src_ptr[0]));
      auto vf32 = _mm256_loadu_ps(tmpbuf);
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
  constexpr int simd_proc_elt = 8;
  auto col_body = col / simd_proc_elt * simd_proc_elt;
  auto col_tail = col % simd_proc_elt;
  float tmpbuf[simd_proc_elt];
  for (int i = 0; i < row; i++) {
    const auto src = src_ptr + i * src_step;
    const auto dst = dst_ptr + i * dst_step;
    int j = 0;
    for (; j < col_body; j += simd_proc_elt) {
      auto vf32 = load_T_fp32(src + j);
      _mm256_storeu_ps(dst + j, vf32);
    }
    if (col_tail > 0) {
      memcpy(tmpbuf, src + j, col_tail * sizeof(src_ptr[0]));
      auto vf32 = load_T_fp32((T*)tmpbuf);
      _mm256_storeu_ps(tmpbuf, vf32);
      memcpy(dst + j, tmpbuf, col_tail * sizeof(dst[0]));
    }
    if (zeropadding && npadding) std::memset(dst + col, 0, npadding);
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE layernorm(const float* srcptr, const float* scaleptr, const float* biasptr, float epsilon,
                                  int norm_size, float* dstptr, float* mean_out, float* mean_square_out,
                                  bool simplified) {
  int constexpr VLen = 8;
  int norm_size8 = utils::padto_le(norm_size, VLen);
  int h = 0;
  __m256 vmean = _mm256_setzero_ps(), vmeansq = _mm256_setzero_ps();
  for (; h < norm_size8; h += VLen) {
    auto tmp = _mm256_loadu_ps(srcptr + h);
    vmean = _mm256_add_ps(vmean, tmp);
    tmp = _mm256_mul_ps(tmp, tmp);
    vmeansq = _mm256_add_ps(vmeansq, tmp);
  }
  float mean = avx2_reduce_ps<AVX2_REDUCE_TYPE::ADD>(vmean);
  float mean_square = avx2_reduce_ps<AVX2_REDUCE_TYPE::ADD>(vmeansq);
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
  auto vm = _mm256_set1_ps(mean);
  float inv_meansq = 1.f / mean_square;
  auto vms = _mm256_set1_ps(inv_meansq);
  h = 0;
  if (simplified) {
    if (scaleptr) {
      for (; h < norm_size8; h += VLen) {
        auto inp = _mm256_loadu_ps(srcptr + h);
        auto scale = _mm256_loadu_ps(scaleptr + h);
        inp = _mm256_mul_ps(inp, scale);
        inp = _mm256_mul_ps(inp, vms);
        _mm256_storeu_ps(dstptr + h, inp);
      }
      for (; h < norm_size; h++) {
        dstptr[h] = srcptr[h] * inv_meansq * scaleptr[h];
      }
    } else {
      for (; h < norm_size8; h += VLen) {
        auto inp = _mm256_loadu_ps(srcptr + h);
        inp = _mm256_mul_ps(inp, vms);
        _mm256_storeu_ps(dstptr + h, inp);
      }
      for (; h < norm_size; h++) {
        dstptr[h] = srcptr[h] * inv_meansq;
      }
    }

  } else {
    if (scaleptr) {
      if (biasptr == nullptr) {
        for (; h < norm_size8; h += VLen) {
          auto inp = _mm256_loadu_ps(srcptr + h);
          auto scale = _mm256_loadu_ps(scaleptr + h);
          inp = _mm256_sub_ps(inp, vm);
          inp = _mm256_mul_ps(inp, scale);
          inp = _mm256_mul_ps(inp, vms);
          _mm256_storeu_ps(dstptr + h, inp);
        }
        for (; h < norm_size; h++) {
          dstptr[h] = (srcptr[h] - mean) * inv_meansq * scaleptr[h];
        }
      } else {
        for (; h < norm_size8; h += VLen) {
          auto inp = _mm256_loadu_ps(srcptr + h);
          auto scale = _mm256_loadu_ps(scaleptr + h);
          inp = _mm256_sub_ps(inp, vm);
          inp = _mm256_mul_ps(inp, vms);
          inp = _mm256_mul_ps(inp, scale);
          auto bias = _mm256_loadu_ps(biasptr + h);
          inp = _mm256_add_ps(inp, bias);
          _mm256_storeu_ps(dstptr + h, inp);
        }
        for (; h < norm_size; h++) {
          dstptr[h] = (srcptr[h] - mean) * inv_meansq * scaleptr[h] + biasptr[h];
        }
      }
    } else {
      for (; h < norm_size8; h += VLen) {
        auto inp = _mm256_loadu_ps(srcptr + h);
        inp = _mm256_sub_ps(inp, vm);
        inp = _mm256_mul_ps(inp, vms);
        _mm256_storeu_ps(dstptr + h, inp);
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

template <BTLA_DTYPE S3_T, typename _DST_T>
inline BTLA_CODE decompress_kblock_s3_s8fp(utils::bit2x4* bit2ptr, utils::bit1x8* bit1ptr, _DST_T* dstptr,
                                           int interleave_n_offset, int unpack_elt, int8_t* tmp, size_t tmpsize) {
  auto head_ignore_num = interleave_n_offset % 128;
  const __m256i lowMask = _mm256_set1_epi8(0x03);
  const __m256i highMask = _mm256_set1_epi8(0x04);
  const __m256i bit1Mask = _mm256_set1_epi32(0x0F);
  const __m256i bit1Shift_1 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
  const __m256i bit1Shift_2 = _mm256_set1_epi32((1 << 23) + (1 << 16) + (1 << 9) + (1 << 2));

  auto bit3_interleave_decompress_pack128 = [&](utils::bit2x4* src1, utils::bit1x8* src2, int8_t* dst) {
    __m256i bit2_data = _mm256_loadu_si256((const __m256i*)src1);
    int32_t* bit1_ptr = reinterpret_cast<int32_t*>(src2);
    for (int i = 0; i < 4; i++) {
      auto bit1x32 = _mm256_set1_epi32(bit1_ptr[i]);
      bit1x32 = _mm256_srlv_epi32(bit1x32, bit1Shift_1);
      bit1x32 = _mm256_and_si256(bit1x32, bit1Mask);
      bit1x32 = _mm256_mullo_epi32(bit1x32, bit1Shift_2);
      bit1x32 = _mm256_and_si256(highMask, bit1x32);

      auto bit2x32 = _mm256_and_si256(lowMask, _mm256_srli_epi16(bit2_data, 2 * i));
      auto res = _mm256_add_epi8(bit1x32, bit2x32);
      res = _mm256_sub_epi8(res, highMask);
      _mm256_storeu_si256((__m256i*)(dst + 32 * i), res);
    }
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

  bestla::kernel::jit::DecompressS3::forward_avx2(bit2ptr + compress_wei_ptr_offset / 4,
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

template <BTLA_DTYPE S2_T, typename _DST_T>
inline BTLA_CODE decompress_kblock_s2_s8fp(utils::bit2x4* bit2ptr, _DST_T* dstptr, int unpack_elt, int8_t* tmp,
                                           size_t tmpsize) {
  int constexpr VBits = 256;
  int constexpr VElt = VBits / 8;
  int i = 0;
  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm256_set_epi64x(*(int64_t*)&mask0, *(int64_t*)&mask0, *(int64_t*)&mask0, *(int64_t*)&mask0);
  auto vshift_y = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm256_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);
  int elt_pad = utils::padto_le(unpack_elt, VElt);
  for (; i < elt_pad; i += VElt) {
    auto vout = unpack_2bits(bit2ptr + i / 4, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
    if (std::is_same_v<_DST_T, int8_t>) {
      _mm256_storeu_si256((__m256i*)(dstptr + i), vout);
    } else {
      _mm256_storeu_si256((__m256i*)tmp, vout);
      for (int j = 0; j < VElt; j++) {
        dstptr[i + j] = tmp[j];
      }
    }
  }
  ref::decompress_kblock_s2_s8fp<S2_T, _DST_T>(bit2ptr + i / 4, dstptr + i, unpack_elt - i, tmp, tmpsize);
  return BTLA_CODE::Success;
}

template <BTLA_DTYPE _S2_T, typename _DST_T, int _PACK_ROW, typename _ST>
static inline BTLA_CODE decompress_kblock_bit2_packrow_fp(utils::bit2x4* bit2ptr, _DST_T* dstptr, int row, int col,
                                                          _ST* scales, int8_t* zero_points, int k_offset, int kblock,
                                                          int NPad, void* tmp, size_t tmpsize) {
  auto unpack_elt = row * col;
  decompress_kblock_s2_s8fp<_S2_T>(bit2ptr, dstptr, unpack_elt, reinterpret_cast<int8_t*>(tmp), tmpsize);
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

inline __m256 poly_scale_2nd_ps(const __m256i z, const __m256 f, const __m256 c0, const __m256 c1, const __m256 c2) {
  const auto y = _mm256_fmadd_ps(_mm256_fmadd_ps(f, c0, c1), f, c2);  // auto y = (f * c0 + c1) * f + c2;
  static const auto mask_exp = _mm256_set1_epi32(0x7f800000);
  static const auto mask_not_exp = _mm256_set1_epi32(~0x7f800000);

  const auto y_exp = _mm256_and_si256(_mm256_castps_si256(y), mask_exp);
  const auto y_not_exp = _mm256_and_si256(_mm256_castps_si256(y), mask_not_exp);

  const auto y_exp_scaled = _mm256_add_epi32(y_exp, _mm256_slli_epi32(z, 23));
  return _mm256_castsi256_ps(_mm256_or_si256(y_not_exp, _mm256_and_si256(y_exp_scaled, mask_exp)));
}

inline __m256 exp_ps_0_1(const __m256 x) {
  static const auto c0 = _mm256_set1_ps(0.240226507f);
  static const auto c1 = _mm256_set1_ps(0.452920674f);
  static const auto c2 = _mm256_set1_ps(0.713483036f);
  static const float v_log2e = std::log2(std::exp(1.f));
  static const auto log2e = _mm256_set1_ps(v_log2e);
  static const auto half = _mm256_set1_ps(.5f);

  static const auto upper_bound = _mm256_set1_ps(88.722838f);   // log(max_positive_float)
  static const auto lower_bound = _mm256_set1_ps(-87.336549f);  // log(min_positive_float)
  __m256 x1 = _mm256_min_ps(x, upper_bound);
  x1 = _mm256_max_ps(x1, lower_bound);

  x1 = _mm256_fmadd_ps(x1, log2e, half);  // auto x1 = x * log2e + _mm256_set1_ps(.5f);
  const auto z = _mm256_floor_ps(x1);
  const auto f = _mm256_sub_ps(x1, z);  // auto f = x1 - z;

  return poly_scale_2nd_ps(_mm256_cvtps_epi32(z), f, c0, c1, c2);
}

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"  // https://stackoverflow.com/a/49216021
#endif
// Interleave 8 xmm vectors of words inplace
static inline std::array<__m128i, 8> tr_x8_word(std::array<__m128i, 8>& src) {  // NOLINT [runtime/references]
  std::array<__m128i, 8> dst;

  for (int i = 0; i < 8; i += 2) {
    dst[i + 0] = _mm_unpacklo_epi16(src[i + 0], src[i + 1]);
    dst[i + 1] = _mm_unpackhi_epi16(src[i + 0], src[i + 1]);
  }
  for (int i = 0; i < 8; i += 4) {
    src[i + 0] = _mm_unpacklo_epi32(dst[i + 0], dst[i + 2]);
    src[i + 1] = _mm_unpackhi_epi32(dst[i + 0], dst[i + 2]);
    src[i + 2] = _mm_unpacklo_epi32(dst[i + 1], dst[i + 3]);
    src[i + 3] = _mm_unpackhi_epi32(dst[i + 1], dst[i + 3]);
  }
  dst[0] = _mm_unpacklo_epi64(src[0], src[4]);
  dst[1] = _mm_unpackhi_epi64(src[0], src[4]);
  dst[2] = _mm_unpacklo_epi64(src[1], src[5]);
  dst[3] = _mm_unpackhi_epi64(src[1], src[5]);
  dst[4] = _mm_unpacklo_epi64(src[2], src[6]);
  dst[5] = _mm_unpackhi_epi64(src[2], src[6]);
  dst[6] = _mm_unpacklo_epi64(src[3], src[7]);
  dst[7] = _mm_unpackhi_epi64(src[3], src[7]);
  return dst;
}

template <int tail>
inline std::array<__m128i, 8> load_fp32_fp16_tr_x8_word(const float* a, size_t lda) {
  static_assert(tail > 0 && tail <= 8, "Unexpected tail value.");
  std::array<__m128i, 8> dst;
  for (int i = 0; i < tail; ++i) {
    dst[i] = _mm256_cvtps_ph(_mm256_loadu_ps(a + i * lda), _MM_FROUND_TO_NEAREST_INT);
  }
  for (int i = tail; i < 8; ++i) dst[i] = _mm_setzero_si128();
  return tr_x8_word(dst);
}
constexpr decltype(load_fp32_fp16_tr_x8_word<1>)* load_fp32_fp16_tr_x8_word_tbl[9]{
    load_fp32_fp16_tr_x8_word<1>, load_fp32_fp16_tr_x8_word<1>, load_fp32_fp16_tr_x8_word<2>,
    load_fp32_fp16_tr_x8_word<3>, load_fp32_fp16_tr_x8_word<4>, load_fp32_fp16_tr_x8_word<5>,
    load_fp32_fp16_tr_x8_word<6>, load_fp32_fp16_tr_x8_word<7>, load_fp32_fp16_tr_x8_word<8>};

template <int tail>
inline std::array<__m128i, 8> load_maskz_fp32_fp16_tr_x8_word(const float* a, size_t lda, __m256i mask) {
  static_assert(tail > 0 && tail <= 8, "Unexpected tail value.");
  std::array<__m128i, 8> dst;
  for (int i = 0; i < tail; ++i) {
    dst[i] = _mm256_cvtps_ph(_mm256_maskload_ps(a + i * lda, mask), _MM_FROUND_TO_NEAREST_INT);
  }
  for (int i = tail; i < 8; ++i) dst[i] = _mm_setzero_si128();
  return tr_x8_word(dst);
}
constexpr decltype(load_maskz_fp32_fp16_tr_x8_word<1>)* load_maskz_fp32_fp16_tr_x8_word_tbl[9]{
    load_maskz_fp32_fp16_tr_x8_word<1>, load_maskz_fp32_fp16_tr_x8_word<1>, load_maskz_fp32_fp16_tr_x8_word<2>,
    load_maskz_fp32_fp16_tr_x8_word<3>, load_maskz_fp32_fp16_tr_x8_word<4>, load_maskz_fp32_fp16_tr_x8_word<5>,
    load_maskz_fp32_fp16_tr_x8_word<6>, load_maskz_fp32_fp16_tr_x8_word<7>, load_maskz_fp32_fp16_tr_x8_word<8>};

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

template <int MTILE, int NReg, int Unroll>
static inline void accumulate_fp32_s8_fp32(const float* Aptr, int lda, int8_t* Bptr, __m256* vacc, __m256* vsca) {
  if constexpr (MTILE == 1) {
    for (int ikk = 0; ikk < Unroll; ikk++) {
      __m256 va = _mm256_set1_ps(*(Aptr + ikk));
      for (int i = 0; i < NReg; i++) {
        auto ftmp = load_s8_fp32(Bptr + i * 8 + ikk * NReg * 8);
        ftmp = _mm256_mul_ps(ftmp, vsca[i]);
        vacc[i] = _mm256_fmadd_ps(va, ftmp, vacc[i]);
      }
    }
  } else {
    for (int ikk = 0; ikk < Unroll; ikk++) {
      __m256 va[MTILE];
      for (int i = 0; i < NReg; i++) {
        auto ftmp = load_s8_fp32(Bptr + i * 8 + ikk * NReg * 8);
        ftmp = _mm256_mul_ps(ftmp, vsca[i]);
        for (int im = 0; im < MTILE; im++) {
          if (i == 0) {
            va[im] = _mm256_set1_ps(*(Aptr + ikk + im * lda));
          }
          vacc[im * NReg + i] = _mm256_fmadd_ps(va[im], ftmp, vacc[im * NReg + i]);
        }
      }
    }
  }
}

template <int MTILE, int NReg, int Unroll>
static inline void accumulate_fp32_s8_fp32(const float* Aptr, int lda, int8_t* Bptr, __m256* vacc_loc) {
  if constexpr (MTILE == 1) {
    for (int ikk = 0; ikk < Unroll; ikk++) {
      __m256 va = _mm256_set1_ps(*(Aptr + ikk));
      for (int i = 0; i < NReg; i++) {
        auto ftmp = load_s8_fp32(Bptr + i * 8 + ikk * NReg * 8);
        vacc_loc[i] = _mm256_fmadd_ps(va, ftmp, vacc_loc[i]);
      }
    }
  } else {
    for (int ikk = 0; ikk < Unroll; ikk++) {
      __m256 va[MTILE];
      for (int i = 0; i < NReg; i++) {
        auto ftmp = load_s8_fp32(Bptr + i * 8 + ikk * NReg * 8);
        for (int im = 0; im < MTILE; im++) {
          if (i == 0) {
            va[im] = _mm256_set1_ps(*(Aptr + ikk + im * lda));
          }
          vacc_loc[im * NReg + i] = _mm256_fmadd_ps(va[im], ftmp, vacc_loc[im * NReg + i]);
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
  int constexpr NReg = NTILE / 8;
  int constexpr MReg = MTILE;
  // Initialize accumulator with zeros
  __m256 acc[NReg * MReg];
  for (int i = 0; i < NReg * MReg; i++) {
    acc[i] = _mm256_setzero_ps();
  }
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm256_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm256_set1_epi8(8);
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;
    __m256 v_b_scale[NReg];
    for (int i = 0; i < NReg; i++) {
      v_b_scale[i] = load_T_fp32(bsptr + i * 8);
    }

    int constexpr Unroll = 4;
    assert((blocksize % 4) == 0);
    assert(tmpsize >= NTILE * Unroll);

    if (B.zpptr) {
      __m256i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;

      for (int i = 0; i < Unroll; i++) {
        memcpy(tmp + i * NTILE, bzptr, NTILE);
      }
      for (int i = 0; i < NReg; i++) {
        bzp[i] = _mm256_loadu_si256((const __m256i*)(tmp + i * 32));
        bzp[i] = _mm256_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += Unroll) {
        for (int i = 0; i < NReg; i++) {
          auto vb = kernel::avx2::unpack_4bits((void*)(b4ptr + i * 16 + (ib * blocksize + ik) * NTILE / 2), vmask);
          vb = _mm256_sub_epi8(vb, bzp[i]);
          _mm256_storeu_si256((__m256i*)(tmp + 32 * i), vb);
        }
        accumulate_fp32_s8_fp32<MTILE, NReg, Unroll>(A + ib * blocksize + ik, lda, tmp, acc, v_b_scale);
      }

    } else {
      for (int ik = 0; ik < blocksize; ik += Unroll) {
        for (int i = 0; i < NReg; i++) {
          auto vb = kernel::avx2::unpack_4bits((void*)(b4ptr + i * 16 + (ib * blocksize + ik) * NTILE / 2), vmask);
          vb = _mm256_sub_epi8(vb, vbias);
          _mm256_storeu_si256((__m256i*)(tmp + 32 * i), vb);
        }
        accumulate_fp32_s8_fp32<MTILE, NReg, Unroll>(A + ib * blocksize + ik, lda, tmp, acc, v_b_scale);
      }
    }
  }

  for (int j = 0; j < MReg; j++) {
    for (int i = 0; i < NReg; i++) {
      _mm256_storeu_ps(C + i * 8 + j * ldc, acc[j * NReg + i]);
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_2bit_fp32_fp32(const float* A, int lda, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  auto b2ptr = (utils::bit2x4*)B.b2ptr;

  int blks = k / blocksize;
  int constexpr NReg = NTILE / 8;
  int constexpr MReg = MTILE;
  // Initialize accumulator with zeros
  __m256 acc[NReg * MReg];
  for (int i = 0; i < NReg * MReg; i++) {
    acc[i] = _mm256_setzero_ps();
  }
  uint64_t mask0 = 0x0303030303030303;
  auto vmask0_y = _mm256_set_epi64x(*(int64_t*)&mask0, *(int64_t*)&mask0, *(int64_t*)&mask0, *(int64_t*)&mask0);
  auto vshift_y = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm256_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);
  auto vbias = _mm256_set1_epi8(2);

  int constexpr KTILE = 1;
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;

    __m256 acc_loc[NReg * MReg];
    for (int i = 0; i < NReg * MReg; i++) {
      acc_loc[i] = _mm256_setzero_ps();
    }
    int constexpr Unroll = 4;
    assert((blocksize % 4) == 0);
    assert(tmpsize >= NTILE * Unroll);

    if (B.zpptr) {
      __m256i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < Unroll; i++) {
        memcpy(tmp + i * NTILE, bzptr, NTILE);
      }
      for (int i = 0; i < NReg; i++) {
        bzp[i] = _mm256_loadu_si256((const __m256i*)(tmp + i * 32));
        bzp[i] = _mm256_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += Unroll) {
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_2bits(b2ptr, vshift_y, vmask0_y, vsfhl_mask_y, vorder_y);
          vb = _mm256_sub_epi8(vb, bzp[i]);
          _mm256_storeu_si256((__m256i*)(tmp + 32 * i), vb);
          b2ptr += 8 * Unroll / 4;
        }
        accumulate_fp32_s8_fp32<MTILE, NReg, Unroll>(A + ib * blocksize + ik, lda, tmp, acc_loc);
      }

    } else {
      for (int ik = 0; ik < blocksize; ik += Unroll) {
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_2bits(b2ptr, vshift_y, vmask0_y, vsfhl_mask_y, vorder_y);
          vb = _mm256_sub_epi8(vb, vbias);
          _mm256_storeu_si256((__m256i*)(tmp + 32 * i), vb);
          b2ptr += 8 * Unroll / 4;
        }
        accumulate_fp32_s8_fp32<MTILE, NReg, Unroll>(A + ib * blocksize + ik, lda, tmp, acc_loc);
      }
    }

    __m256 v_b_scale[NReg];
    for (int i = 0; i < NReg; i++) {
      v_b_scale[i] = load_T_fp32(bsptr + i * 8);
    }
    for (int im = 0; im < MTILE; im++) {
      for (int in = 0; in < NReg; in++) {
        acc[im * NReg + in] = _mm256_fmadd_ps(acc_loc[im * NReg + in], v_b_scale[in], acc[im * NReg + in]);
      }
    }
  }

  for (int j = 0; j < MReg; j++) {
    for (int i = 0; i < NReg; i++) {
      _mm256_storeu_ps(C + i * 8 + j * ldc, acc[j * NReg + i]);
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_1bit_fp32_fp32(const float* A, int lda, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  auto b1ptr = (utils::bit1x8*)B.b1ptr;

  int blks = k / blocksize;
  int constexpr NReg = NTILE / 8;
  int constexpr MReg = MTILE;
  int constexpr FullRange = 1 << (1 - 1);
  __m256 acc[NReg * MReg];
  for (int i = 0; i < NReg * MReg; i++) {
    acc[i] = _mm256_setzero_ps();
  }
  auto vbias = _mm256_set1_epi8(FullRange);

  const __m256i highMask = _mm256_set1_epi8(0x04);
  const __m256i bit1Mask = _mm256_set1_epi32(0x0F);
  const __m256i bit1Shift_1 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
  const __m256i bit1Shift_2 = _mm256_set1_epi32((1 << 23) + (1 << 16) + (1 << 9) + (1 << 2));
  int constexpr KTILE = 1;
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;

    __m256 acc_loc[NReg * MReg];
    for (int i = 0; i < NReg * MReg; i++) {
      acc_loc[i] = _mm256_setzero_ps();
    }
    int constexpr Unroll = 4;
    assert((blocksize % 4) == 0);
    assert(tmpsize >= NTILE * Unroll);

    if (B.zpptr) {
      __m256i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < Unroll; i++) {
        memcpy(tmp + i * NTILE, bzptr, NTILE);
      }
      for (int i = 0; i < NReg; i++) {
        bzp[i] = _mm256_loadu_si256((const __m256i*)(tmp + i * 32));
        bzp[i] = _mm256_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += Unroll) {
        for (int i = 0; i < NReg; i++) {
          auto vb1 = unpack_1bits(b1ptr, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
          vb1 = _mm256_srli_epi32(vb1, 2);
          vb1 = _mm256_sub_epi8(vb1, bzp[i]);
          _mm256_storeu_si256((__m256i*)(tmp + 32 * i), vb1);
          b1ptr += 8 * Unroll / 8;
        }
        accumulate_fp32_s8_fp32<MTILE, NReg, Unroll>(A + ib * blocksize + ik, lda, tmp, acc_loc);
      }

    } else {
      for (int ik = 0; ik < blocksize; ik += Unroll) {
        for (int i = 0; i < NReg; i++) {
          auto vb1 = unpack_1bits(b1ptr, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
          vb1 = _mm256_srli_epi32(vb1, 2);
          vb1 = _mm256_sub_epi8(vb1, vbias);
          _mm256_storeu_si256((__m256i*)(tmp + 32 * i), vb1);
          b1ptr += 8 * Unroll / 8;
        }
        accumulate_fp32_s8_fp32<MTILE, NReg, Unroll>(A + ib * blocksize + ik, lda, tmp, acc_loc);
      }
    }

    __m256 v_b_scale[NReg];
    for (int i = 0; i < NReg; i++) {
      v_b_scale[i] = load_T_fp32(bsptr + i * 8);
    }
    for (int im = 0; im < MTILE; im++) {
      for (int in = 0; in < NReg; in++) {
        acc[im * NReg + in] = _mm256_fmadd_ps(acc_loc[im * NReg + in], v_b_scale[in], acc[im * NReg + in]);
      }
    }
  }

  for (int j = 0; j < MReg; j++) {
    for (int i = 0; i < NReg; i++) {
      _mm256_storeu_ps(C + i * 8 + j * ldc, acc[j * NReg + i]);
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_3bit_fp32_fp32(const float* A, int lda, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  auto b2ptr = (utils::bit2x4*)B.b2ptr;
  auto b1ptr = (utils::bit1x8*)B.b1ptr;

  int blks = k / blocksize;
  int constexpr FullRange = 1 << (3 - 1);
  int constexpr NReg = NTILE / 8;
  int constexpr MReg = MTILE;
  // Initialize accumulator with zeros
  __m256 acc[NReg * MReg];
  for (int i = 0; i < NReg * MReg; i++) {
    acc[i] = _mm256_setzero_ps();
  }

  uint64_t mask0 = 0x0303030303030303;
  auto vmask0_y = _mm256_set_epi64x(*(int64_t*)&mask0, *(int64_t*)&mask0, *(int64_t*)&mask0, *(int64_t*)&mask0);
  auto vshift_y = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm256_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);
  auto vbias = _mm256_set1_epi8(FullRange);

  const __m256i highMask = _mm256_set1_epi8(0x04);
  const __m256i bit1Mask = _mm256_set1_epi32(0x0F);
  const __m256i bit1Shift_1 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
  const __m256i bit1Shift_2 = _mm256_set1_epi32((1 << 23) + (1 << 16) + (1 << 9) + (1 << 2));
  int constexpr KTILE = 1;
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;

    __m256 acc_loc[NReg * MReg];
    for (int i = 0; i < NReg * MReg; i++) {
      acc_loc[i] = _mm256_setzero_ps();
    }
    int constexpr Unroll = 4;
    assert((blocksize % 4) == 0);
    assert(tmpsize >= NTILE * Unroll);

    if (B.zpptr) {
      __m256i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < Unroll; i++) {
        memcpy(tmp + i * NTILE, bzptr, NTILE);
      }
      for (int i = 0; i < NReg; i++) {
        bzp[i] = _mm256_loadu_si256((const __m256i*)(tmp + i * 32));
        bzp[i] = _mm256_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += Unroll) {
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_2bits(b2ptr, vshift_y, vmask0_y, vsfhl_mask_y, vorder_y);
          auto vb1 = unpack_1bits(b1ptr, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
          vb = _mm256_or_si256(vb, vb1);
          vb = _mm256_sub_epi8(vb, bzp[i]);
          _mm256_storeu_si256((__m256i*)(tmp + 32 * i), vb);
          b2ptr += 8 * Unroll / 4;
          b1ptr += 8 * Unroll / 8;
        }
        accumulate_fp32_s8_fp32<MTILE, NReg, Unroll>(A + ib * blocksize + ik, lda, tmp, acc_loc);
      }

    } else {
      for (int ik = 0; ik < blocksize; ik += Unroll) {
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_2bits(b2ptr, vshift_y, vmask0_y, vsfhl_mask_y, vorder_y);
          auto vb1 = unpack_1bits(b1ptr, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
          vb = _mm256_or_si256(vb, vb1);
          vb = _mm256_sub_epi8(vb, vbias);
          _mm256_storeu_si256((__m256i*)(tmp + 32 * i), vb);
          b2ptr += 8 * Unroll / 4;
          b1ptr += 8 * Unroll / 8;
        }
        accumulate_fp32_s8_fp32<MTILE, NReg, Unroll>(A + ib * blocksize + ik, lda, tmp, acc_loc);
      }
    }

    __m256 v_b_scale[NReg];
    for (int i = 0; i < NReg; i++) {
      v_b_scale[i] = load_T_fp32(bsptr + i * 8);
    }
    for (int im = 0; im < MTILE; im++) {
      for (int in = 0; in < NReg; in++) {
        acc[im * NReg + in] = _mm256_fmadd_ps(acc_loc[im * NReg + in], v_b_scale[in], acc[im * NReg + in]);
      }
    }
  }

  for (int j = 0; j < MReg; j++) {
    for (int i = 0; i < NReg; i++) {
      _mm256_storeu_ps(C + i * 8 + j * ldc, acc[j * NReg + i]);
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_5bit_fp32_fp32(const float* A, int lda, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  auto b4ptr = (utils::bit4x2*)B.b4ptr;
  auto b1ptr = (utils::bit1x8*)B.b1ptr;

  int blks = k / blocksize;
  int constexpr NReg = NTILE / 8;
  int constexpr MReg = MTILE;
  // Initialize accumulator with zeros
  __m256 acc[NReg * MReg];
  for (int i = 0; i < NReg * MReg; i++) {
    acc[i] = _mm256_setzero_ps();
  }

  int constexpr FullRange = 1 << (5 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm256_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm256_set1_epi8(FullRange);

  const __m256i highMask = _mm256_set1_epi8(0x04);
  const __m256i bit1Mask = _mm256_set1_epi32(0x0F);
  const __m256i bit1Shift_1 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
  const __m256i bit1Shift_2 = _mm256_set1_epi32((1 << 23) + (1 << 16) + (1 << 9) + (1 << 2));
  int constexpr KTILE = 1;
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;

    __m256 acc_loc[NReg * MReg];
    for (int i = 0; i < NReg * MReg; i++) {
      acc_loc[i] = _mm256_setzero_ps();
    }
    int constexpr Unroll = 4;
    assert((blocksize % 4) == 0);
    assert(tmpsize >= NTILE * Unroll);

    if (B.zpptr) {
      __m256i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < Unroll; i++) {
        memcpy(tmp + i * NTILE, bzptr, NTILE);
      }
      for (int i = 0; i < NReg; i++) {
        bzp[i] = _mm256_loadu_si256((const __m256i*)(tmp + i * 32));
        bzp[i] = _mm256_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += Unroll) {
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_4bits(b4ptr, vmask);
          auto vb1 = unpack_1bits(b1ptr, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
          vb1 = _mm256_slli_epi32(vb1, 2);
          vb = _mm256_or_si256(vb, vb1);
          vb = _mm256_sub_epi8(vb, bzp[i]);
          _mm256_storeu_si256((__m256i*)(tmp + 32 * i), vb);
          b4ptr += 8 * Unroll / 2;
          b1ptr += 8 * Unroll / 8;
        }
        accumulate_fp32_s8_fp32<MTILE, NReg, Unroll>(A + ib * blocksize + ik, lda, tmp, acc_loc);
      }

    } else {
      for (int ik = 0; ik < blocksize; ik += Unroll) {
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_4bits(b4ptr, vmask);
          auto vb1 = unpack_1bits(b1ptr, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
          vb1 = _mm256_slli_epi32(vb1, 2);
          vb = _mm256_or_si256(vb, vb1);
          vb = _mm256_sub_epi8(vb, vbias);
          _mm256_storeu_si256((__m256i*)(tmp + 32 * i), vb);
          b4ptr += 8 * Unroll / 2;
          b1ptr += 8 * Unroll / 8;
        }
        accumulate_fp32_s8_fp32<MTILE, NReg, Unroll>(A + ib * blocksize + ik, lda, tmp, acc_loc);
      }
    }

    __m256 v_b_scale[NReg];
    for (int i = 0; i < NReg; i++) {
      v_b_scale[i] = load_T_fp32(bsptr + i * 8);
    }
    for (int im = 0; im < MTILE; im++) {
      for (int in = 0; in < NReg; in++) {
        acc[im * NReg + in] = _mm256_fmadd_ps(acc_loc[im * NReg + in], v_b_scale[in], acc[im * NReg + in]);
      }
    }
  }

  for (int j = 0; j < MReg; j++) {
    for (int i = 0; i < NReg; i++) {
      _mm256_storeu_ps(C + i * 8 + j * ldc, acc[j * NReg + i]);
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_6bit_fp32_fp32(const float* A, int lda, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  auto b4ptr = (utils::bit4x2*)B.b4ptr;
  auto b2ptr = (utils::bit2x4*)B.b2ptr;

  int blks = k / blocksize;
  int constexpr NReg = NTILE / 8;
  int constexpr MReg = MTILE;
  // Initialize accumulator with zeros
  __m256 acc[NReg * MReg];
  for (int i = 0; i < NReg * MReg; i++) {
    acc[i] = _mm256_setzero_ps();
  }

  int constexpr FullRange = 1 << (6 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm256_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm256_set1_epi8(FullRange);

  uint32_t mask0 = 0x03030303;
  auto vmask0 = _mm256_set1_epi32(*(int32_t*)&mask0);
  auto vshift_y = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm256_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);
  int constexpr KTILE = 1;
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;

    __m256 acc_loc[NReg * MReg];
    for (int i = 0; i < NReg * MReg; i++) {
      acc_loc[i] = _mm256_setzero_ps();
    }
    int constexpr Unroll = 4;
    assert((blocksize % 4) == 0);
    assert(tmpsize >= NTILE * Unroll);

    if (B.zpptr) {
      __m256i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < Unroll; i++) {
        memcpy(tmp + i * NTILE, bzptr, NTILE);
      }
      for (int i = 0; i < NReg; i++) {
        bzp[i] = _mm256_loadu_si256((const __m256i*)(tmp + i * 32));
        bzp[i] = _mm256_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += Unroll) {
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_4bits(b4ptr, vmask);
          auto vb1 = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
          vb1 = _mm256_slli_epi32(vb1, 4);
          vb = _mm256_or_si256(vb, vb1);
          vb = _mm256_sub_epi8(vb, bzp[i]);
          _mm256_storeu_si256((__m256i*)(tmp + 32 * i), vb);
          b4ptr += 8 * Unroll / 2;
          b2ptr += 8 * Unroll / 4;
        }
        accumulate_fp32_s8_fp32<MTILE, NReg, Unroll>(A + ib * blocksize + ik, lda, tmp, acc_loc);
      }

    } else {
      for (int ik = 0; ik < blocksize; ik += Unroll) {
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_4bits(b4ptr, vmask);
          auto vb1 = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
          vb1 = _mm256_slli_epi32(vb1, 4);
          vb = _mm256_or_si256(vb, vb1);
          vb = _mm256_sub_epi8(vb, vbias);
          _mm256_storeu_si256((__m256i*)(tmp + 32 * i), vb);
          b4ptr += 8 * Unroll / 2;
          b2ptr += 8 * Unroll / 4;
        }
        accumulate_fp32_s8_fp32<MTILE, NReg, Unroll>(A + ib * blocksize + ik, lda, tmp, acc_loc);
      }
    }

    __m256 v_b_scale[NReg];
    for (int i = 0; i < NReg; i++) {
      v_b_scale[i] = load_T_fp32(bsptr + i * 8);
    }
    for (int im = 0; im < MTILE; im++) {
      for (int in = 0; in < NReg; in++) {
        acc[im * NReg + in] = _mm256_fmadd_ps(acc_loc[im * NReg + in], v_b_scale[in], acc[im * NReg + in]);
      }
    }
  }

  for (int j = 0; j < MReg; j++) {
    for (int i = 0; i < NReg; i++) {
      _mm256_storeu_ps(C + i * 8 + j * ldc, acc[j * NReg + i]);
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

  int blks = k / blocksize;
  int constexpr NReg = NTILE / 8;
  int constexpr MReg = MTILE;
  // Initialize accumulator with zeros
  __m256 acc[NReg * MReg];
  for (int i = 0; i < NReg * MReg; i++) {
    acc[i] = _mm256_setzero_ps();
  }

  int constexpr FullRange = 1 << (7 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm256_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm256_set1_epi8(FullRange);

  uint32_t mask0 = 0x03030303;
  auto vmask0 = _mm256_set1_epi32(*(int32_t*)&mask0);
  auto vshift_y = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm256_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);

  const __m256i highMask = _mm256_set1_epi8(0x04);
  const __m256i bit1Mask = _mm256_set1_epi32(0x0F);
  const __m256i bit1Shift_1 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
  const __m256i bit1Shift_2 = _mm256_set1_epi32((1 << 23) + (1 << 16) + (1 << 9) + (1 << 2));
  int constexpr KTILE = 1;
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;

    __m256 acc_loc[NReg * MReg];
    for (int i = 0; i < NReg * MReg; i++) {
      acc_loc[i] = _mm256_setzero_ps();
    }
    int constexpr Unroll = 4;
    assert((blocksize % 4) == 0);
    assert(tmpsize >= NTILE * Unroll);

    if (B.zpptr) {
      __m256i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < Unroll; i++) {
        memcpy(tmp + i * NTILE, bzptr, NTILE);
      }
      for (int i = 0; i < NReg; i++) {
        bzp[i] = _mm256_loadu_si256((const __m256i*)(tmp + i * 32));
        bzp[i] = _mm256_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += Unroll) {
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_4bits(b4ptr, vmask);
          auto vb1 = unpack_1bits(b1ptr, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
          auto vb2 = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
          vb1 = _mm256_slli_epi32(vb1, 4);
          vb2 = _mm256_slli_epi32(vb2, 4);
          vb = _mm256_or_si256(vb, vb1);
          vb = _mm256_or_si256(vb, vb2);
          vb = _mm256_sub_epi8(vb, bzp[i]);
          _mm256_storeu_si256((__m256i*)(tmp + 32 * i), vb);
          b4ptr += 8 * Unroll / 2;
          b1ptr += 8 * Unroll / 8;
          b2ptr += 8 * Unroll / 4;
        }
        accumulate_fp32_s8_fp32<MTILE, NReg, Unroll>(A + ib * blocksize + ik, lda, tmp, acc_loc);
      }

    } else {
      for (int ik = 0; ik < blocksize; ik += Unroll) {
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_4bits(b4ptr, vmask);
          auto vb1 = unpack_1bits(b1ptr, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
          auto vb2 = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
          vb1 = _mm256_slli_epi32(vb1, 4);
          vb2 = _mm256_slli_epi32(vb2, 4);
          vb = _mm256_or_si256(vb, vb1);
          vb = _mm256_or_si256(vb, vb2);
          vb = _mm256_sub_epi8(vb, vbias);
          _mm256_storeu_si256((__m256i*)(tmp + 32 * i), vb);
          b4ptr += 8 * Unroll / 2;
          b1ptr += 8 * Unroll / 8;
          b2ptr += 8 * Unroll / 4;
        }
        accumulate_fp32_s8_fp32<MTILE, NReg, Unroll>(A + ib * blocksize + ik, lda, tmp, acc_loc);
      }
    }

    __m256 v_b_scale[NReg];
    for (int i = 0; i < NReg; i++) {
      v_b_scale[i] = load_T_fp32(bsptr + i * 8);
    }
    for (int im = 0; im < MTILE; im++) {
      for (int in = 0; in < NReg; in++) {
        acc[im * NReg + in] = _mm256_fmadd_ps(acc_loc[im * NReg + in], v_b_scale[in], acc[im * NReg + in]);
      }
    }
  }

  for (int j = 0; j < MReg; j++) {
    for (int i = 0; i < NReg; i++) {
      _mm256_storeu_ps(C + i * 8 + j * ldc, acc[j * NReg + i]);
    }
  }
  return BTLA_CODE::Success;
}

static inline __m256i _mm256_dpbusd_avx2_epi32(__m256i& c, const __m256i& a, const __m256i& b) {
  const __m256i dot2 = _mm256_maddubs_epi16(a, b);
  const __m256i ones = _mm256_set1_epi16(1);
  const __m256i sum4 = _mm256_madd_epi16(ones, dot2);
  return _mm256_add_epi32(c, sum4);
}

template <typename ScaleT, int NReg, int MTILE>
static inline void gemv_dequant_s32fp32(const float* asptr, int ldzp, const ScaleT* bsptr, __m256i* iacc,
                                        __m256* facc) {
  __m256 v_a_scale[MTILE];
  for (int im = 0; im < MTILE; im++) {
    v_a_scale[im] = _mm256_set1_ps(*(asptr + im * ldzp));
  }

  for (int i = 0; i < NReg; i++) {
    __m256 v_b_scale = load_T_fp32(bsptr + i * 8);
    for (int im = 0; im < MTILE; im++) {
      auto vtmp = _mm256_mul_ps(v_a_scale[im], v_b_scale);
      auto tmp = _mm256_cvtepi32_ps(iacc[im * NReg + i]);
      facc[im * NReg + i] = _mm256_fmadd_ps(tmp, vtmp, facc[im * NReg + i]);
    }
  }
}

template <int NReg, int MReg>
static inline void gemv_remove_zp(const uint8_t* azptr, int ldzp, __m256i* iacc, __m256i* bacc) {
  if constexpr (MReg == 1) {
    auto zp = int(azptr[0]);
    __m256i v_a_zp = _mm256_set1_epi32(zp);
    for (int in = 0; in < NReg; in++) {
      auto vtmp = _mm256_mullo_epi32(v_a_zp, bacc[in]);
      iacc[in] = _mm256_sub_epi32(iacc[in], vtmp);
    }
  } else {
    __m256i v_a_zp[MReg];
    for (int im = 0; im < MReg; im++) {
      auto zp = int(azptr[im * ldzp]);
      v_a_zp[im] = _mm256_set1_epi32(zp);
      for (int in = 0; in < NReg; in++) {
        auto vtmp = _mm256_mullo_epi32(v_a_zp[im], bacc[in]);
        iacc[im * NReg + in] = _mm256_sub_epi32(iacc[im * NReg + in], vtmp);
      }
    }
  }
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_4bit_u8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  auto& a8ptr = A.aptr;
  auto& b4ptr = B.b4ptr;
  auto& asptr = A.sptr;
  auto& azptr = A.zpptr;

  int blks = k / blocksize;
  int constexpr NReg = NTILE / 8;
  int constexpr MReg = MTILE;
  // Initialize accumulator with zeros
  __m256 acc[NReg * MReg];
  for (int i = 0; i < NReg * MReg; i++) {
    acc[i] = _mm256_setzero_ps();
  }
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm256_set1_epi32(*reinterpret_cast<int*>(&mask));
  const __m256i onesu8 = _mm256_set1_epi8(1);
  const __m256i vbias = _mm256_set1_epi8(8);
  const auto vindex = _mm256_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0);

  for (int ib = 0; ib < blks; ib += 1) {
    __m256i iacc[NReg * MReg];
    __m256i bacc[NReg];
    for (int i = 0; i < NReg * MReg; i++) {
      iacc[i] = _mm256_setzero_si256();
    }
    for (int i = 0; i < NReg; i++) {
      bacc[i] = _mm256_setzero_si256();
    }
    if (B.zpptr) {
      __m256i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < NReg; i++) {
        bzp[i] = load_zp_epi8_broadcast_epi32(bzptr + i * 8, vindex);
        bzp[i] = _mm256_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += 4) {
        if constexpr (MTILE == 1) {
          __m256i va = _mm256_set1_epi32(*(int*)(a8ptr + ib * blocksize + ik));

          for (int i = 0; i < NReg; i++) {
            auto vb = kernel::avx2::unpack_4bits((void*)(b4ptr + i * 16 + (ib * blocksize + ik) * NTILE / 2), vmask);
            vb = _mm256_sub_epi8(vb, bzp[i]);
            bacc[i] = _mm256_dpbusd_avx2_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm256_dpbusd_avx2_epi32(iacc[i], va, vb);
          }
        } else {
          __m256i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm256_set1_epi32(*(int*)(a8ptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb = kernel::avx2::unpack_4bits((void*)(b4ptr + i * 16 + (ib * blocksize + ik) * NTILE / 2), vmask);
            vb = _mm256_sub_epi8(vb, bzp[i]);
            bacc[i] = _mm256_dpbusd_avx2_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm256_dpbusd_avx2_epi32(iacc[j * NReg + i], va[j], vb);
            }
          }
        }
      }
    } else {
      for (int ik = 0; ik < blocksize; ik += 4) {
        if constexpr (MTILE == 1) {
          __m256i va = _mm256_set1_epi32(*(int*)(a8ptr + ib * blocksize + ik));
          for (int i = 0; i < NReg; i++) {
            auto vb = kernel::avx2::unpack_4bits((void*)(b4ptr + i * 16 + (ib * blocksize + ik) * NTILE / 2), vmask);
            vb = _mm256_sub_epi8(vb, vbias);
            bacc[i] = _mm256_dpbusd_avx2_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm256_dpbusd_avx2_epi32(iacc[i], va, vb);
          }
        } else {
          __m256i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm256_set1_epi32(*(int*)(a8ptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb = kernel::avx2::unpack_4bits((void*)(b4ptr + i * 16 + (ib * blocksize + ik) * NTILE / 2), vmask);
            vb = _mm256_sub_epi8(vb, vbias);
            bacc[i] = _mm256_dpbusd_avx2_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm256_dpbusd_avx2_epi32(iacc[j * NReg + i], va[j], vb);
            }
          }
        }
      }
    }

    gemv_remove_zp<NReg, MTILE>(A.zpptr + ib, A.ldzp, iacc, bacc);
    gemv_dequant_s32fp32<ScaleT, NReg, MTILE>(A.sptr + ib, A.ldzp, B.sptr + ib * B.ldzp, iacc, acc);
  }

  for (int j = 0; j < MReg; j++) {
    for (int i = 0; i < NReg; i++) {
      _mm256_storeu_ps(C + i * 8 + j * ldc, acc[j * NReg + i]);
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_1bit_u8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  auto b1ptr = reinterpret_cast<utils::bit1x8*>(B.b1ptr);

  int blks = k / blocksize;
  int constexpr FullRange = 1 << (1 - 1);
  int constexpr NReg = NTILE / 8;
  int constexpr MReg = MTILE;
  __m256 acc[NReg * MReg];
  for (int i = 0; i < NReg * MReg; i++) {
    acc[i] = _mm256_setzero_ps();
  }
  const __m256i onesu8 = _mm256_set1_epi8(1);
  const __m256i vbias = _mm256_set1_epi8(FullRange);
  const auto vindex = _mm256_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0);

  const __m256i highMask = _mm256_set1_epi8(0x04);
  const __m256i bit1Mask = _mm256_set1_epi32(0x0F);
  const __m256i bit1Shift_1 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
  const __m256i bit1Shift_2 = _mm256_set1_epi32((1 << 23) + (1 << 16) + (1 << 9) + (1 << 2));
  int constexpr KTILE = 4;
  for (int ib = 0; ib < blks; ib += 1) {
    __m256i iacc[NReg * MReg];
    __m256i bacc[NReg];
    for (int i = 0; i < NReg * MReg; i++) {
      iacc[i] = _mm256_setzero_si256();
    }
    for (int i = 0; i < NReg; i++) {
      bacc[i] = _mm256_setzero_si256();
    }
    if (B.zpptr) {
      __m256i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < NReg; i++) {
        bzp[i] = load_zp_epi8_broadcast_epi32(bzptr + i * 8, vindex);
        bzp[i] = _mm256_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        if constexpr (MTILE == 1) {
          __m256i va = _mm256_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik));
          for (int i = 0; i < NReg; i++) {
            auto vb1 = unpack_1bits(b1ptr, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
            vb1 = _mm256_srli_epi32(vb1, 2);
            vb1 = _mm256_sub_epi8(vb1, bzp[i]);
            bacc[i] = _mm256_dpbusd_avx2_epi32(bacc[i], onesu8, vb1);
            iacc[i] = _mm256_dpbusd_avx2_epi32(iacc[i], va, vb1);
            b1ptr += 8 * KTILE / 8;
          }
        } else {
          __m256i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm256_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb1 = unpack_1bits(b1ptr, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
            vb1 = _mm256_srli_epi32(vb1, 2);
            vb1 = _mm256_sub_epi8(vb1, bzp[i]);
            bacc[i] = _mm256_dpbusd_avx2_epi32(bacc[i], onesu8, vb1);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm256_dpbusd_avx2_epi32(iacc[j * NReg + i], va[j], vb1);
            }
            b1ptr += 8 * KTILE / 8;
          }
        }
      }
    } else {
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        if constexpr (MTILE == 1) {
          __m256i va = _mm256_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik));
          for (int i = 0; i < NReg; i++) {
            auto vb1 = unpack_1bits(b1ptr, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
            vb1 = _mm256_srli_epi32(vb1, 2);
            vb1 = _mm256_sub_epi8(vb1, vbias);
            bacc[i] = _mm256_dpbusd_avx2_epi32(bacc[i], onesu8, vb1);
            iacc[i] = _mm256_dpbusd_avx2_epi32(iacc[i], va, vb1);
            b1ptr += 8 * KTILE / 8;
          }
        } else {
          __m256i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm256_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb1 = unpack_1bits(b1ptr, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
            vb1 = _mm256_srli_epi32(vb1, 2);
            vb1 = _mm256_sub_epi8(vb1, vbias);
            bacc[i] = _mm256_dpbusd_avx2_epi32(bacc[i], onesu8, vb1);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm256_dpbusd_avx2_epi32(iacc[j * NReg + i], va[j], vb1);
            }
            b1ptr += 8 * KTILE / 8;
          }
        }
      }
    }

    gemv_remove_zp<NReg, MReg>(A.zpptr + ib, A.ldzp, iacc, bacc);
    gemv_dequant_s32fp32<ScaleT, NReg, MTILE>(A.sptr + ib, A.ldzp, B.sptr + ib * B.ldzp, iacc, acc);
  }

  for (int j = 0; j < MReg; j++) {
    for (int i = 0; i < NReg; i++) {
      _mm256_storeu_ps(C + i * 8 + j * ldc, acc[j * NReg + i]);
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
  int constexpr FullRange = 1 << (3 - 1);
  int constexpr NReg = NTILE / 8;
  int constexpr MReg = MTILE;
  __m256 acc[NReg * MReg];
  for (int i = 0; i < NReg * MReg; i++) {
    acc[i] = _mm256_setzero_ps();
  }
  uint64_t mask0 = 0x0303030303030303;
  auto vmask0_y = _mm256_set_epi64x(*(int64_t*)&mask0, *(int64_t*)&mask0, *(int64_t*)&mask0, *(int64_t*)&mask0);
  auto vshift_y = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm256_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);
  const __m256i onesu8 = _mm256_set1_epi8(1);
  const __m256i vbias = _mm256_set1_epi8(FullRange);
  const auto vindex = _mm256_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0);

  const __m256i highMask = _mm256_set1_epi8(0x04);
  const __m256i bit1Mask = _mm256_set1_epi32(0x0F);
  const __m256i bit1Shift_1 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
  const __m256i bit1Shift_2 = _mm256_set1_epi32((1 << 23) + (1 << 16) + (1 << 9) + (1 << 2));
  int constexpr KTILE = 4;
  for (int ib = 0; ib < blks; ib += 1) {
    __m256i iacc[NReg * MReg];
    __m256i bacc[NReg];
    for (int i = 0; i < NReg * MReg; i++) {
      iacc[i] = _mm256_setzero_si256();
    }
    for (int i = 0; i < NReg; i++) {
      bacc[i] = _mm256_setzero_si256();
    }
    if (B.zpptr) {
      __m256i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < NReg; i++) {
        bzp[i] = load_zp_epi8_broadcast_epi32(bzptr + i * 8, vindex);
        bzp[i] = _mm256_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        if constexpr (MTILE == 1) {
          __m256i va = _mm256_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik));
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_2bits(b2ptr, vshift_y, vmask0_y, vsfhl_mask_y, vorder_y);
            auto vb1 = unpack_1bits(b1ptr, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
            vb = _mm256_or_si256(vb, vb1);
            vb = _mm256_sub_epi8(vb, bzp[i]);
            bacc[i] = _mm256_dpbusd_avx2_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm256_dpbusd_avx2_epi32(iacc[i], va, vb);
            b2ptr += 8 * KTILE / 4;
            b1ptr += 8 * KTILE / 8;
          }
        } else {
          __m256i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm256_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_2bits(b2ptr, vshift_y, vmask0_y, vsfhl_mask_y, vorder_y);
            auto vb1 = unpack_1bits(b1ptr, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
            vb = _mm256_or_si256(vb, vb1);
            vb = _mm256_sub_epi8(vb, bzp[i]);
            bacc[i] = _mm256_dpbusd_avx2_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm256_dpbusd_avx2_epi32(iacc[j * NReg + i], va[j], vb);
            }
            b2ptr += 8 * KTILE / 4;
            b1ptr += 8 * KTILE / 8;
          }
        }
      }
    } else {
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        if constexpr (MTILE == 1) {
          __m256i va = _mm256_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik));
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_2bits(b2ptr, vshift_y, vmask0_y, vsfhl_mask_y, vorder_y);
            auto vb1 = unpack_1bits(b1ptr, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
            vb = _mm256_or_si256(vb, vb1);
            vb = _mm256_sub_epi8(vb, vbias);
            bacc[i] = _mm256_dpbusd_avx2_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm256_dpbusd_avx2_epi32(iacc[i], va, vb);

            b2ptr += 8 * KTILE / 4;
            b1ptr += 8 * KTILE / 8;
          }
        } else {
          __m256i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm256_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_2bits(b2ptr, vshift_y, vmask0_y, vsfhl_mask_y, vorder_y);
            auto vb1 = unpack_1bits(b1ptr, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
            vb = _mm256_or_si256(vb, vb1);
            vb = _mm256_sub_epi8(vb, vbias);
            bacc[i] = _mm256_dpbusd_avx2_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm256_dpbusd_avx2_epi32(iacc[j * NReg + i], va[j], vb);
            }
            b2ptr += 8 * KTILE / 4;
            b1ptr += 8 * KTILE / 8;
          }
        }
      }
    }

    gemv_remove_zp<NReg, MReg>(A.zpptr + ib, A.ldzp, iacc, bacc);
    gemv_dequant_s32fp32<ScaleT, NReg, MTILE>(A.sptr + ib, A.ldzp, B.sptr + ib * B.ldzp, iacc, acc);
  }

  for (int j = 0; j < MReg; j++) {
    for (int i = 0; i < NReg; i++) {
      _mm256_storeu_ps(C + i * 8 + j * ldc, acc[j * NReg + i]);
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
  int constexpr NReg = NTILE / 8;
  int constexpr MReg = MTILE;
  __m256 acc[NReg * MReg];
  for (int i = 0; i < NReg * MReg; i++) {
    acc[i] = _mm256_setzero_ps();
  }

  int constexpr FullRange = 1 << (5 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm256_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm256_set1_epi8(FullRange);

  const __m256i onesu8 = _mm256_set1_epi8(1);
  const auto vindex = _mm256_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0);

  const __m256i highMask = _mm256_set1_epi8(0x04);
  const __m256i bit1Mask = _mm256_set1_epi32(0x0F);
  const __m256i bit1Shift_1 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
  const __m256i bit1Shift_2 = _mm256_set1_epi32((1 << 23) + (1 << 16) + (1 << 9) + (1 << 2));
  int constexpr KTILE = 4;
  for (int ib = 0; ib < blks; ib += 1) {
    __m256i iacc[NReg * MReg];
    __m256i bacc[NReg];
    for (int i = 0; i < NReg * MReg; i++) {
      iacc[i] = _mm256_setzero_si256();
    }
    for (int i = 0; i < NReg; i++) {
      bacc[i] = _mm256_setzero_si256();
    }
    if (B.zpptr) {
      __m256i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < NReg; i++) {
        bzp[i] = load_zp_epi8_broadcast_epi32(bzptr + i * 8, vindex);
        bzp[i] = _mm256_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        if constexpr (MTILE == 1) {
          __m256i va = _mm256_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik));
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_4bits(b4ptr, vmask);
            auto vb1 = unpack_1bits(b1ptr, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
            vb1 = _mm256_slli_epi32(vb1, 2);
            vb = _mm256_or_si256(vb, vb1);
            vb = _mm256_sub_epi8(vb, bzp[i]);
            bacc[i] = _mm256_dpbusd_avx2_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm256_dpbusd_avx2_epi32(iacc[i], va, vb);
            b4ptr += 8 * KTILE / 2;
            b1ptr += 8 * KTILE / 8;
          }
        } else {
          __m256i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm256_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_4bits(b4ptr, vmask);
            auto vb1 = unpack_1bits(b1ptr, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
            vb1 = _mm256_slli_epi32(vb1, 2);
            vb = _mm256_or_si256(vb, vb1);
            vb = _mm256_sub_epi8(vb, bzp[i]);
            bacc[i] = _mm256_dpbusd_avx2_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm256_dpbusd_avx2_epi32(iacc[j * NReg + i], va[j], vb);
            }
            b4ptr += 8 * KTILE / 2;
            b1ptr += 8 * KTILE / 8;
          }
        }
      }
    } else {
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        if constexpr (MTILE == 1) {
          __m256i va = _mm256_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik));
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_4bits(b4ptr, vmask);
            auto vb1 = unpack_1bits(b1ptr, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
            vb1 = _mm256_slli_epi32(vb1, 2);
            vb = _mm256_or_si256(vb, vb1);
            vb = _mm256_sub_epi8(vb, vbias);
            bacc[i] = _mm256_dpbusd_avx2_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm256_dpbusd_avx2_epi32(iacc[i], va, vb);
            b4ptr += 8 * KTILE / 2;
            b1ptr += 8 * KTILE / 8;
          }
        } else {
          __m256i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm256_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_4bits(b4ptr, vmask);
            auto vb1 = unpack_1bits(b1ptr, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
            vb1 = _mm256_slli_epi32(vb1, 2);
            vb = _mm256_or_si256(vb, vb1);
            vb = _mm256_sub_epi8(vb, vbias);
            bacc[i] = _mm256_dpbusd_avx2_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm256_dpbusd_avx2_epi32(iacc[j * NReg + i], va[j], vb);
            }
            b4ptr += 8 * KTILE / 2;
            b1ptr += 8 * KTILE / 8;
          }
        }
      }
    }

    gemv_remove_zp<NReg, MReg>(A.zpptr + ib, A.ldzp, iacc, bacc);
    gemv_dequant_s32fp32<ScaleT, NReg, MTILE>(A.sptr + ib, A.ldzp, B.sptr + ib * B.ldzp, iacc, acc);
  }

  for (int j = 0; j < MReg; j++) {
    for (int i = 0; i < NReg; i++) {
      _mm256_storeu_ps(C + i * 8 + j * ldc, acc[j * NReg + i]);
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
  int constexpr NReg = NTILE / 8;
  int constexpr MReg = MTILE;
  __m256 acc[NReg * MReg];
  for (int i = 0; i < NReg * MReg; i++) {
    acc[i] = _mm256_setzero_ps();
  }

  int constexpr FullRange = 1 << (6 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm256_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm256_set1_epi8(FullRange);

  const __m256i onesu8 = _mm256_set1_epi8(1);
  const auto vindex = _mm256_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0);

  uint32_t mask0 = 0x03030303;
  auto vmask0 = _mm256_set1_epi32(*(int32_t*)&mask0);
  auto vshift_y = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm256_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);
  int constexpr KTILE = 4;
  for (int ib = 0; ib < blks; ib += 1) {
    __m256i iacc[NReg * MReg];
    __m256i bacc[NReg];
    for (int i = 0; i < NReg * MReg; i++) {
      iacc[i] = _mm256_setzero_si256();
    }
    for (int i = 0; i < NReg; i++) {
      bacc[i] = _mm256_setzero_si256();
    }
    if (B.zpptr) {
      __m256i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < NReg; i++) {
        bzp[i] = load_zp_epi8_broadcast_epi32(bzptr + i * 8, vindex);
        bzp[i] = _mm256_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        if constexpr (MTILE == 1) {
          __m256i va = _mm256_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik));
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_4bits(b4ptr, vmask);
            auto vb1 = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
            vb1 = _mm256_slli_epi32(vb1, 4);
            vb = _mm256_or_si256(vb, vb1);
            vb = _mm256_sub_epi8(vb, bzp[i]);
            bacc[i] = _mm256_dpbusd_avx2_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm256_dpbusd_avx2_epi32(iacc[i], va, vb);
            b4ptr += 8 * KTILE / 2;
            b2ptr += 8 * KTILE / 4;
          }
        } else {
          __m256i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm256_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_4bits(b4ptr, vmask);
            auto vb1 = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
            vb1 = _mm256_slli_epi32(vb1, 4);
            vb = _mm256_or_si256(vb, vb1);
            vb = _mm256_sub_epi8(vb, bzp[i]);
            bacc[i] = _mm256_dpbusd_avx2_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm256_dpbusd_avx2_epi32(iacc[j * NReg + i], va[j], vb);
            }
            b4ptr += 8 * KTILE / 2;
            b2ptr += 8 * KTILE / 4;
          }
        }
      }
    } else {
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        if constexpr (MTILE == 1) {
          __m256i va = _mm256_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik));
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_4bits(b4ptr, vmask);
            auto vb1 = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
            vb1 = _mm256_slli_epi32(vb1, 4);
            vb = _mm256_or_si256(vb, vb1);
            vb = _mm256_sub_epi8(vb, vbias);
            bacc[i] = _mm256_dpbusd_avx2_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm256_dpbusd_avx2_epi32(iacc[i], va, vb);
            b4ptr += 8 * KTILE / 2;
            b2ptr += 8 * KTILE / 4;
          }
        } else {
          __m256i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm256_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_4bits(b4ptr, vmask);
            auto vb1 = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
            vb1 = _mm256_slli_epi32(vb1, 4);
            vb = _mm256_or_si256(vb, vb1);
            vb = _mm256_sub_epi8(vb, vbias);
            bacc[i] = _mm256_dpbusd_avx2_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm256_dpbusd_avx2_epi32(iacc[j * NReg + i], va[j], vb);
            }
            b4ptr += 8 * KTILE / 2;
            b2ptr += 8 * KTILE / 4;
          }
        }
      }
    }

    gemv_remove_zp<NReg, MReg>(A.zpptr + ib, A.ldzp, iacc, bacc);
    gemv_dequant_s32fp32<ScaleT, NReg, MTILE>(A.sptr + ib, A.ldzp, B.sptr + ib * B.ldzp, iacc, acc);
  }

  for (int j = 0; j < MReg; j++) {
    for (int i = 0; i < NReg; i++) {
      _mm256_storeu_ps(C + i * 8 + j * ldc, acc[j * NReg + i]);
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
  int constexpr NReg = NTILE / 8;
  int constexpr MReg = MTILE;
  __m256 acc[NReg * MReg];
  for (int i = 0; i < NReg * MReg; i++) {
    acc[i] = _mm256_setzero_ps();
  }

  int constexpr FullRange = 1 << (7 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm256_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm256_set1_epi8(FullRange);

  const __m256i onesu8 = _mm256_set1_epi8(1);
  const auto vindex = _mm256_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0);

  const __m256i highMask = _mm256_set1_epi8(0x04);
  const __m256i bit1Mask = _mm256_set1_epi32(0x0F);
  const __m256i bit1Shift_1 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
  const __m256i bit1Shift_2 = _mm256_set1_epi32((1 << 23) + (1 << 16) + (1 << 9) + (1 << 2));

  uint32_t mask0 = 0x03030303;
  auto vmask0 = _mm256_set1_epi32(*(int32_t*)&mask0);
  auto vshift_y = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm256_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);
  int constexpr KTILE = 4;
  for (int ib = 0; ib < blks; ib += 1) {
    __m256i iacc[NReg * MReg];
    __m256i bacc[NReg];
    for (int i = 0; i < NReg * MReg; i++) {
      iacc[i] = _mm256_setzero_si256();
    }
    for (int i = 0; i < NReg; i++) {
      bacc[i] = _mm256_setzero_si256();
    }
    if (B.zpptr) {
      __m256i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < NReg; i++) {
        bzp[i] = load_zp_epi8_broadcast_epi32(bzptr + i * 8, vindex);
        bzp[i] = _mm256_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        if constexpr (MTILE == 1) {
          __m256i va = _mm256_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik));
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_4bits(b4ptr, vmask);
            auto vb1 = unpack_1bits(b1ptr, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
            auto vb2 = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
            vb1 = _mm256_slli_epi32(vb1, 4);
            vb2 = _mm256_slli_epi32(vb2, 4);
            vb = _mm256_or_si256(vb, vb1);
            vb = _mm256_or_si256(vb, vb2);
            vb = _mm256_sub_epi8(vb, bzp[i]);
            bacc[i] = _mm256_dpbusd_avx2_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm256_dpbusd_avx2_epi32(iacc[i], va, vb);
            b4ptr += 8 * KTILE / 2;
            b1ptr += 8 * KTILE / 8;
            b2ptr += 8 * KTILE / 4;
          }
        } else {
          __m256i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm256_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_4bits(b4ptr, vmask);
            auto vb1 = unpack_1bits(b1ptr, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
            auto vb2 = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
            vb1 = _mm256_slli_epi32(vb1, 4);
            vb2 = _mm256_slli_epi32(vb2, 4);
            vb = _mm256_or_si256(vb, vb1);
            vb = _mm256_or_si256(vb, vb2);
            vb = _mm256_sub_epi8(vb, bzp[i]);
            bacc[i] = _mm256_dpbusd_avx2_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm256_dpbusd_avx2_epi32(iacc[j * NReg + i], va[j], vb);
            }
            b4ptr += 8 * KTILE / 2;
            b1ptr += 8 * KTILE / 8;
            b2ptr += 8 * KTILE / 4;
          }
        }
      }
    } else {
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        if constexpr (MTILE == 1) {
          __m256i va = _mm256_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik));
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_4bits(b4ptr, vmask);
            auto vb1 = unpack_1bits(b1ptr, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
            auto vb2 = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
            vb1 = _mm256_slli_epi32(vb1, 4);
            vb2 = _mm256_slli_epi32(vb2, 4);
            vb = _mm256_or_si256(vb, vb1);
            vb = _mm256_or_si256(vb, vb2);
            vb = _mm256_sub_epi8(vb, vbias);
            bacc[i] = _mm256_dpbusd_avx2_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm256_dpbusd_avx2_epi32(iacc[i], va, vb);
            b4ptr += 8 * KTILE / 2;
            b1ptr += 8 * KTILE / 8;
            b2ptr += 8 * KTILE / 4;
          }
        } else {
          __m256i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm256_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_4bits(b4ptr, vmask);
            auto vb1 = unpack_1bits(b1ptr, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
            auto vb2 = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
            vb1 = _mm256_slli_epi32(vb1, 4);
            vb2 = _mm256_slli_epi32(vb2, 4);
            vb = _mm256_or_si256(vb, vb1);
            vb = _mm256_or_si256(vb, vb2);
            vb = _mm256_sub_epi8(vb, vbias);
            bacc[i] = _mm256_dpbusd_avx2_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm256_dpbusd_avx2_epi32(iacc[j * NReg + i], va[j], vb);
            }
            b4ptr += 8 * KTILE / 2;
            b1ptr += 8 * KTILE / 8;
            b2ptr += 8 * KTILE / 4;
          }
        }
      }
    }

    gemv_remove_zp<NReg, MReg>(A.zpptr + ib, A.ldzp, iacc, bacc);
    gemv_dequant_s32fp32<ScaleT, NReg, MTILE>(A.sptr + ib, A.ldzp, B.sptr + ib * B.ldzp, iacc, acc);
  }

  for (int j = 0; j < MReg; j++) {
    for (int i = 0; i < NReg; i++) {
      _mm256_storeu_ps(C + i * 8 + j * ldc, acc[j * NReg + i]);
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_2bit_u8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  auto b2ptr = reinterpret_cast<utils::bit2x4*>(B.b2ptr);

  int blks = k / blocksize;
  int constexpr NReg = NTILE / 8;
  int constexpr MReg = MTILE;
  __m256 acc[NReg * MReg];
  for (int i = 0; i < NReg * MReg; i++) {
    acc[i] = _mm256_setzero_ps();
  }
  uint64_t mask0 = 0x0303030303030303;
  auto vmask0_y = _mm256_set_epi64x(*(int64_t*)&mask0, *(int64_t*)&mask0, *(int64_t*)&mask0, *(int64_t*)&mask0);
  auto vshift_y = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm256_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);
  const __m256i onesu8 = _mm256_set1_epi8(1);
  const __m256i vbias = _mm256_set1_epi8(2);
  const auto vindex = _mm256_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0);
  int constexpr KTILE = 4;
  for (int ib = 0; ib < blks; ib += 1) {
    __m256i iacc[NReg * MReg];
    __m256i bacc[NReg];
    for (int i = 0; i < NReg * MReg; i++) {
      iacc[i] = _mm256_setzero_si256();
    }
    for (int i = 0; i < NReg; i++) {
      bacc[i] = _mm256_setzero_si256();
    }
    if (B.zpptr) {
      __m256i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < NReg; i++) {
        bzp[i] = load_zp_epi8_broadcast_epi32(bzptr + i * 8, vindex);
        bzp[i] = _mm256_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        if constexpr (MTILE == 1) {
          __m256i va = _mm256_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik));
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_2bits(b2ptr, vshift_y, vmask0_y, vsfhl_mask_y, vorder_y);
            vb = _mm256_sub_epi8(vb, bzp[i]);
            bacc[i] = _mm256_dpbusd_avx2_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm256_dpbusd_avx2_epi32(iacc[i], va, vb);
            b2ptr += 8 * KTILE / 4;
          }
        } else {
          __m256i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm256_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_2bits(b2ptr, vshift_y, vmask0_y, vsfhl_mask_y, vorder_y);
            vb = _mm256_sub_epi8(vb, bzp[i]);
            bacc[i] = _mm256_dpbusd_avx2_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm256_dpbusd_avx2_epi32(iacc[j * NReg + i], va[j], vb);
            }
            b2ptr += 8 * KTILE / 4;
          }
        }
      }
    } else {
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        if constexpr (MTILE == 1) {
          __m256i va = _mm256_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik));
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_2bits(b2ptr, vshift_y, vmask0_y, vsfhl_mask_y, vorder_y);
            vb = _mm256_sub_epi8(vb, vbias);
            bacc[i] = _mm256_dpbusd_avx2_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm256_dpbusd_avx2_epi32(iacc[i], va, vb);
            b2ptr += 8 * KTILE / 4;
          }
        } else {
          __m256i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm256_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_2bits(b2ptr, vshift_y, vmask0_y, vsfhl_mask_y, vorder_y);
            vb = _mm256_sub_epi8(vb, vbias);
            bacc[i] = _mm256_dpbusd_avx2_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm256_dpbusd_avx2_epi32(iacc[j * NReg + i], va[j], vb);
            }
            b2ptr += 8 * KTILE / 4;
          }
        }
      }
    }

    gemv_remove_zp<NReg, MReg>(A.zpptr + ib, A.ldzp, iacc, bacc);
    gemv_dequant_s32fp32<ScaleT, NReg, MTILE>(A.sptr + ib, A.ldzp, B.sptr + ib * B.ldzp, iacc, acc);
  }

  for (int j = 0; j < MReg; j++) {
    for (int i = 0; i < NReg; i++) {
      _mm256_storeu_ps(C + i * 8 + j * ldc, acc[j * NReg + i]);
    }
  }
  return BTLA_CODE::Success;
}

template <typename T>
static inline BTLA_CODE mul(const T* src0ptr, const T* src1ptr, T* dstptr, size_t size) {
  int constexpr VLen = 8;
  size_t velt = utils::padto_le(size, VLen);
  size_t i = 0;
  auto vfunc = [&]() {
    auto v0 = load_T_fp32(src0ptr + i);
    auto v1 = load_T_fp32(src1ptr + i);
    auto out = _mm256_mul_ps(v0, v1);
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
static inline BTLA_CODE add(const T* src0ptr, const T* src1ptr, T* dstptr, size_t size) {
  int constexpr VLen = 8;
  size_t velt = utils::padto_le(size, VLen);
  size_t i = 0;
  auto vfunc = [&]() {
    auto v0 = load_T_fp32(src0ptr + i);
    auto v1 = load_T_fp32(src1ptr + i);
    auto out = _mm256_add_ps(v0, v1);
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

  const auto v_scale = _mm256_set1_ps(scale);
  const auto v_seq7 = _mm256_loadu_ps(seq15);
  const auto alibi_slope = _mm256_set1_ps(_alibi_slope);
  const auto alibi_base = _mm256_mul_ps(alibi_slope, _mm256_add_ps(v_seq7, _mm256_set1_ps(N_offset)));
  const auto alibi_step = _mm256_set1_ps(_alibi_slope * 8);
  const auto infinity_neg = _mm256_set1_ps(-INFINITY);
  for (int i = 0; i < M; ++i) {
    auto alibi_curr = alibi_base;
    const auto N_unmasked = std::min(N, padding_type == 0   ? INT32_MAX
                                        : padding_type == 1 ? i + M_offset - N_offset + causal_offset + 1
                                                            : causal_offset - N_offset);

    const auto v_mask = _mm256_load_si256(reinterpret_cast<const __m256i*>(mask8[N_unmasked % 8]));
    int j = 0;
    auto v_max = infinity_neg;
    for (; j < N_unmasked - 7; j += 8) {
      const auto xs = _mm256_fmadd_ps(v_scale, _mm256_loadu_ps(src + i * src_step + j), alibi_curr);
      v_max = _mm256_max_ps(v_max, xs);
      _mm256_storeu_ps(dst + i * ld_dst + j, xs);
      if constexpr (HAS_ALIBI) alibi_curr = _mm256_add_ps(alibi_curr, alibi_step);
    }
    if (j < N_unmasked) {
      const auto xs = _mm256_fmadd_ps(v_scale, _mm256_maskload_ps(src + i * src_step + j, v_mask), alibi_curr);
      const auto masked_xs = _mm256_blendv_ps(infinity_neg, xs, _mm256_castsi256_ps(v_mask));
      v_max = _mm256_max_ps(v_max, masked_xs);
      _mm256_storeu_ps(dst + i * ld_dst + j, xs);
      if constexpr (HAS_ALIBI) alibi_curr = _mm256_add_ps(alibi_curr, alibi_step);
      j += 8;
    }
    alignas(32) float dst_tmp[8];
    _mm256_store_ps(dst_tmp, v_max);
    for (int ii = 0; ii < 8; ++ii) dst_max[i] = std::max(dst_max[i], dst_tmp[ii]);
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE weight_cvt_fp16_fp32_n24(const utils::fp16* B, int ldb, bool is_padded, float* dst_ptr,
                                                 int dst_step, int k_size, int n_size, int k_offset, int n_offset,
                                                 void* tmpcache, size_t cachesize) {
  assert(is_padded);
  const auto src = B + k_offset * 24 + n_offset * ldb;
  assert(n_size <= 24);
  assert(n_offset % 24 == 0);
  if (n_size == 24) {
    constexpr auto n_size = 24;
    for (int i = 0; i < k_size; ++i) {
      for (int j = 0; j < n_size; j += 8) {
        const auto cur_src = src + i * 24 + j;
        const auto cur_dst = dst_ptr + i * 24 + j;
        const auto src = load_T_fp32(cur_src);
        _mm256_store_ps(cur_dst, src);
      }
    }
  } else {
    for (int i = 0; i < k_size; ++i) {
      for (int j = 0; j < n_size; j += 8) {
        const auto cur_src = src + i * 24 + j;
        const auto cur_dst = dst_ptr + i * 24 + j;
        const auto src = load_T_fp32(cur_src);
        _mm256_store_ps(cur_dst, src);
      }
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE inplace_precompute_max_softmax_fp32_fp32(int m_size, int n_size, int n_pad_size, bool is_causal,
                                                                 float* src, float* dst, const float* s_max,
                                                                 float* expsum, int ld_src, int ld_dst) {
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

  for (int ii = 0; ii < m_size; ++ii) {
    const auto i_src = src + ii * ld_src;
    const auto i_dst = dst + ii * ld_dst;
    const auto curr_n_size = n_size + (is_causal ? ii : 0);
    const auto v_mask = _mm256_load_si256(reinterpret_cast<const __m256i*>(mask8[curr_n_size % 8]));
    {  // subtract max
      const auto row_max = _mm256_set1_ps(s_max[ii]);
      for (int jj = 0; jj < curr_n_size; jj += 8) {  // should be fine to do extra work on idx >= curr_n_size
        _mm256_storeu_ps(i_src + jj, _mm256_sub_ps(_mm256_loadu_ps(i_src + jj), row_max));
      }
    }
    auto v_sum = _mm256_setzero_ps();
    {  // exp & sum
      int jj = 0;
      for (; jj < utils::padto_le(curr_n_size, 8); jj += 8) {
        const auto v_exp = kernel::avx2::exp_ps_0_1(_mm256_loadu_ps(i_src + jj));
        v_sum = _mm256_add_ps(v_sum, v_exp);
        _mm256_storeu_ps(i_src + jj, v_exp);
      }
      if (jj < curr_n_size) {
        const auto v_exp = kernel::avx2::exp_ps_0_1(_mm256_loadu_ps(i_src + jj));  // should be fine to load extra
        v_sum = _mm256_add_ps(v_sum, _mm256_and_ps(v_exp, _mm256_castsi256_ps(v_mask)));
        _mm256_storeu_ps(i_src + jj, v_exp);  // should be fine to store some extra
      }

      alignas(32) float sum_tmp[8];
      _mm256_store_ps(sum_tmp, v_sum);
      expsum[ii] = 0.f;
      for (int iii = 0; iii < 8; ++iii) expsum[ii] += sum_tmp[iii];
      v_sum = _mm256_set1_ps(expsum[ii]);
    }
    {  // scale & store
      int jj = 0;
      for (; jj < utils::padto_le(curr_n_size, 8); jj += 8) {
        _mm256_store_ps(i_dst + jj, _mm256_div_ps(_mm256_loadu_ps(i_src + jj), v_sum));
      }
      if (jj < curr_n_size) {
        const auto quotient = _mm256_div_ps(_mm256_loadu_ps(i_src + jj), v_sum);
        _mm256_store_ps(i_dst + jj, _mm256_and_ps(quotient, _mm256_castsi256_ps(v_mask)));
        jj += 8;
      }
      if (jj < n_pad_size) memset(i_dst + jj, 0, sizeof(float) * (n_pad_size - jj));
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
}  // namespace avx2
}  // namespace kernel
}  // namespace bestla
