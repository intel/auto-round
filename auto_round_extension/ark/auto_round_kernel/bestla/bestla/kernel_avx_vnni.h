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
#include "kernel_avx2.h"

namespace bestla {
namespace kernel {
namespace avx2 {
#if CompileAVXVNNI()
#if defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "fma", "f16c", "avxvnni")
#elif defined(ICX)
#pragma clang attribute push(__attribute__((target("avx,avx2,fma,avxvnni"))), apply_to = function)
#endif

namespace vnni {

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
            bacc[i] = _mm256_dpbusd_avx_epi32(bacc[i], onesu8, vb1);
            iacc[i] = _mm256_dpbusd_avx_epi32(iacc[i], va, vb1);
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
            bacc[i] = _mm256_dpbusd_avx_epi32(bacc[i], onesu8, vb1);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm256_dpbusd_avx_epi32(iacc[j * NReg + i], va[j], vb1);
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
            bacc[i] = _mm256_dpbusd_avx_epi32(bacc[i], onesu8, vb1);
            iacc[i] = _mm256_dpbusd_avx_epi32(iacc[i], va, vb1);
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
            bacc[i] = _mm256_dpbusd_avx_epi32(bacc[i], onesu8, vb1);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm256_dpbusd_avx_epi32(iacc[j * NReg + i], va[j], vb1);
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
static inline BTLA_CODE gemv_1bit_s8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
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
    for (int i = 0; i < NReg * MReg; i++) {
      iacc[i] = _mm256_setzero_si256();
    }
    if (B.zpptr) {
      __m256i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < NReg; i++) {
        bzp[i] = load_zp_epi8_broadcast_epi32(bzptr + i * 8, vindex);
        bzp[i] = _mm256_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        __m256i va[MReg];
        for (int i = 0; i < MReg; i++) {
          va[i] = _mm256_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
        }
        for (int i = 0; i < NReg; i++) {
          auto vb1 = unpack_1bits(b1ptr, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
          vb1 = _mm256_srli_epi32(vb1, 2);
          vb1 = _mm256_sub_epi8(vb1, bzp[i]);
          for (int j = 0; j < MReg; j++) {
            auto vsb = _mm256_sign_epi8(vb1, va[j]);
            auto vabsa = _mm256_sign_epi8(va[j], va[j]);
            iacc[j * NReg + i] = _mm256_dpbusd_avx_epi32(iacc[j * NReg + i], vabsa, vsb);
          }
          b1ptr += 8 * KTILE / 8;
        }
      }
    } else {
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        __m256i va[MReg];
        for (int i = 0; i < MReg; i++) {
          va[i] = _mm256_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
        }
        for (int i = 0; i < NReg; i++) {
          auto vb1 = unpack_1bits(b1ptr, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
          vb1 = _mm256_srli_epi32(vb1, 2);
          vb1 = _mm256_sub_epi8(vb1, vbias);
          for (int j = 0; j < MReg; j++) {
            auto vsb = _mm256_sign_epi8(vb1, va[j]);
            auto vabsa = _mm256_sign_epi8(va[j], va[j]);
            iacc[j * NReg + i] = _mm256_dpbusd_avx_epi32(iacc[j * NReg + i], vabsa, vsb);
          }
          b1ptr += 8 * KTILE / 8;
        }
      }
    }

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
static inline BTLA_CODE gemv_4bit_u8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  auto& a8ptr = A.aptr;
  auto& b4ptr = B.b4ptr;
  auto& asptr = A.sptr;
  auto& azptr = A.zpptr;

  int blks = k / blocksize;
  int constexpr FullRange = 1 << (4 - 1);
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
  const __m256i vbias = _mm256_set1_epi8(FullRange);
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
            bacc[i] = _mm256_dpbusd_avx_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm256_dpbusd_avx_epi32(iacc[i], va, vb);
          }
        } else {
          __m256i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm256_set1_epi32(*(int*)(a8ptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb = kernel::avx2::unpack_4bits((void*)(b4ptr + i * 16 + (ib * blocksize + ik) * NTILE / 2), vmask);
            vb = _mm256_sub_epi8(vb, bzp[i]);
            bacc[i] = _mm256_dpbusd_avx_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm256_dpbusd_avx_epi32(iacc[j * NReg + i], va[j], vb);
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
            bacc[i] = _mm256_dpbusd_avx_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm256_dpbusd_avx_epi32(iacc[i], va, vb);
          }
        } else {
          __m256i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm256_set1_epi32(*(int*)(a8ptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb = kernel::avx2::unpack_4bits((void*)(b4ptr + i * 16 + (ib * blocksize + ik) * NTILE / 2), vmask);
            vb = _mm256_sub_epi8(vb, vbias);
            bacc[i] = _mm256_dpbusd_avx_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm256_dpbusd_avx_epi32(iacc[j * NReg + i], va[j], vb);
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
      _mm256_storeu_ps(C + i * 8 + j * ldc, acc[j * NReg + i]);
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_4bit_s8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  auto& a8ptr = A.aptr;
  auto& b4ptr = B.b4ptr;
  auto& asptr = A.sptr;

  int blks = k / blocksize;
  int constexpr FullRange = 1 << (4 - 1);
  int constexpr NReg = NTILE / 8;
  int constexpr MReg = MTILE;
  __m256 acc[NReg * MReg];
  for (int i = 0; i < NReg * MReg; i++) {
    acc[i] = _mm256_setzero_ps();
  }
  const __m256i vbias = _mm256_set1_epi8(FullRange);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm256_set1_epi32(*reinterpret_cast<int*>(&mask));
  const auto vindex = _mm256_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0);
  for (int ib = 0; ib < blks; ib += 1) {
    __m256i iacc[NReg * MReg];
    for (int i = 0; i < NReg * MReg; i++) {
      iacc[i] = _mm256_setzero_si256();
    }
    if (B.zpptr) {
      __m256i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < NReg; i++) {
        bzp[i] = load_zp_epi8_broadcast_epi32(bzptr + i * 8, vindex);
        bzp[i] = _mm256_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += 4) {
        __m256i va[MReg];
        for (int i = 0; i < MReg; i++) {
          va[i] = _mm256_set1_epi32(*(int*)(a8ptr + ib * blocksize + ik + i * A.lda));
        }
        for (int i = 0; i < NReg; i++) {
          auto vb = kernel::avx2::unpack_4bits((void*)(b4ptr + i * 16 + (ib * blocksize + ik) * NTILE / 2), vmask);
          vb = _mm256_sub_epi8(vb, bzp[i]);
          for (int j = 0; j < MReg; j++) {
            auto vsb = _mm256_sign_epi8(vb, va[j]);
            auto vabsa = _mm256_sign_epi8(va[j], va[j]);
            iacc[j * NReg + i] = _mm256_dpbusd_avx_epi32(iacc[j * NReg + i], vabsa, vsb);
          }
        }
      }
    } else {
      for (int ik = 0; ik < blocksize; ik += 4) {
        __m256i va[MReg];
        for (int i = 0; i < MReg; i++) {
          va[i] = _mm256_set1_epi32(*(int*)(a8ptr + ib * blocksize + ik + i * A.lda));
        }
        for (int i = 0; i < NReg; i++) {
          auto vb = kernel::avx2::unpack_4bits((void*)(b4ptr + i * 16 + (ib * blocksize + ik) * NTILE / 2), vmask);
          vb = _mm256_sub_epi8(vb, vbias);
          for (int j = 0; j < MReg; j++) {
            auto vsb = _mm256_sign_epi8(vb, va[j]);
            auto vabsa = _mm256_sign_epi8(va[j], va[j]);
            iacc[j * NReg + i] = _mm256_dpbusd_avx_epi32(iacc[j * NReg + i], vabsa, vsb);
          }
        }
      }
    }

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
  int constexpr FullRange = 1 << (2 - 1);
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
            bacc[i] = _mm256_dpbusd_avx_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm256_dpbusd_avx_epi32(iacc[i], va, vb);
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
            bacc[i] = _mm256_dpbusd_avx_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm256_dpbusd_avx_epi32(iacc[j * NReg + i], va[j], vb);
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
            bacc[i] = _mm256_dpbusd_avx_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm256_dpbusd_avx_epi32(iacc[i], va, vb);
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
            bacc[i] = _mm256_dpbusd_avx_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm256_dpbusd_avx_epi32(iacc[j * NReg + i], va[j], vb);
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

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_2bit_s8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  auto b2ptr = reinterpret_cast<utils::bit2x4*>(B.b2ptr);

  int blks = k / blocksize;
  int constexpr FullRange = 1 << (2 - 1);
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
  int constexpr KTILE = 4;
  for (int ib = 0; ib < blks; ib += 1) {
    __m256i iacc[NReg * MReg];
    for (int i = 0; i < NReg * MReg; i++) {
      iacc[i] = _mm256_setzero_si256();
    }

    if (B.zpptr) {
      __m256i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < NReg; i++) {
        bzp[i] = load_zp_epi8_broadcast_epi32(bzptr + i * 8, vindex);
        bzp[i] = _mm256_add_epi8(vbias, bzp[i]);
      }
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        __m256i va[MReg];
        for (int i = 0; i < MReg; i++) {
          va[i] = _mm256_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
        }
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_2bits(b2ptr, vshift_y, vmask0_y, vsfhl_mask_y, vorder_y);
          vb = _mm256_sub_epi8(vb, bzp[i]);
          for (int j = 0; j < MReg; j++) {
            auto vsb = _mm256_sign_epi8(vb, va[j]);
            auto vabsa = _mm256_sign_epi8(va[j], va[j]);
            iacc[j * NReg + i] = _mm256_dpbusd_avx_epi32(iacc[j * NReg + i], vabsa, vsb);
          }
          b2ptr += 8 * KTILE / 4;
        }
      }
    } else {
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        __m256i va[MReg];
        for (int i = 0; i < MReg; i++) {
          va[i] = _mm256_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
        }
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_2bits(b2ptr, vshift_y, vmask0_y, vsfhl_mask_y, vorder_y);
          vb = _mm256_sub_epi8(vb, vbias);
          for (int j = 0; j < MReg; j++) {
            auto vsb = _mm256_sign_epi8(vb, va[j]);
            auto vabsa = _mm256_sign_epi8(va[j], va[j]);
            iacc[j * NReg + i] = _mm256_dpbusd_avx_epi32(iacc[j * NReg + i], vabsa, vsb);
          }
          b2ptr += 8 * KTILE / 4;
        }
      }
    }
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
            bacc[i] = _mm256_dpbusd_avx_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm256_dpbusd_avx_epi32(iacc[i], va, vb);
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
            bacc[i] = _mm256_dpbusd_avx_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm256_dpbusd_avx_epi32(iacc[j * NReg + i], va[j], vb);
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
            bacc[i] = _mm256_dpbusd_avx_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm256_dpbusd_avx_epi32(iacc[i], va, vb);

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
            bacc[i] = _mm256_dpbusd_avx_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm256_dpbusd_avx_epi32(iacc[j * NReg + i], va[j], vb);
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
static inline BTLA_CODE gemv_3bit_s8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
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
    for (int i = 0; i < NReg * MReg; i++) {
      iacc[i] = _mm256_setzero_si256();
    }
    if (B.zpptr) {
      __m256i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < NReg; i++) {
        bzp[i] = load_zp_epi8_broadcast_epi32(bzptr + i * 8, vindex);
        bzp[i] = _mm256_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        __m256i va[MReg];
        for (int i = 0; i < MReg; i++) {
          va[i] = _mm256_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
        }
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_2bits(b2ptr, vshift_y, vmask0_y, vsfhl_mask_y, vorder_y);
          auto vb1 = unpack_1bits(b1ptr, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
          vb = _mm256_or_si256(vb, vb1);
          vb = _mm256_sub_epi8(vb, bzp[i]);
          for (int j = 0; j < MReg; j++) {
            auto vsb = _mm256_sign_epi8(vb, va[j]);
            auto vabsa = _mm256_sign_epi8(va[j], va[j]);
            iacc[j * NReg + i] = _mm256_dpbusd_avx_epi32(iacc[j * NReg + i], vabsa, vsb);
          }
          b2ptr += 8 * KTILE / 4;
          b1ptr += 8 * KTILE / 8;
        }
      }
    } else {
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        __m256i va[MReg];
        for (int i = 0; i < MReg; i++) {
          va[i] = _mm256_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
        }
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_2bits(b2ptr, vshift_y, vmask0_y, vsfhl_mask_y, vorder_y);
          auto vb1 = unpack_1bits(b1ptr, bit1Shift_1, bit1Mask, bit1Shift_2, highMask);
          vb = _mm256_or_si256(vb, vb1);
          vb = _mm256_sub_epi8(vb, vbias);
          for (int j = 0; j < MReg; j++) {
            auto vsb = _mm256_sign_epi8(vb, va[j]);
            auto vabsa = _mm256_sign_epi8(va[j], va[j]);
            iacc[j * NReg + i] = _mm256_dpbusd_avx_epi32(iacc[j * NReg + i], vabsa, vsb);
          }
          b2ptr += 8 * KTILE / 4;
          b1ptr += 8 * KTILE / 8;
        }
      }
    }

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
            bacc[i] = _mm256_dpbusd_avx_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm256_dpbusd_avx_epi32(iacc[i], va, vb);
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
            bacc[i] = _mm256_dpbusd_avx_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm256_dpbusd_avx_epi32(iacc[j * NReg + i], va[j], vb);
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
            bacc[i] = _mm256_dpbusd_avx_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm256_dpbusd_avx_epi32(iacc[i], va, vb);

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
            bacc[i] = _mm256_dpbusd_avx_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm256_dpbusd_avx_epi32(iacc[j * NReg + i], va[j], vb);
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
static inline BTLA_CODE gemv_5bit_s8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
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
    for (int i = 0; i < NReg * MReg; i++) {
      iacc[i] = _mm256_setzero_si256();
    }
    if (B.zpptr) {
      __m256i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < NReg; i++) {
        bzp[i] = load_zp_epi8_broadcast_epi32(bzptr + i * 8, vindex);
        bzp[i] = _mm256_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += KTILE) {
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
          for (int j = 0; j < MReg; j++) {
            auto vsb = _mm256_sign_epi8(vb, va[j]);
            auto vabsa = _mm256_sign_epi8(va[j], va[j]);
            iacc[j * NReg + i] = _mm256_dpbusd_avx_epi32(iacc[j * NReg + i], vabsa, vsb);
          }
          b4ptr += 8 * KTILE / 2;
          b1ptr += 8 * KTILE / 8;
        }
      }
    } else {
      for (int ik = 0; ik < blocksize; ik += KTILE) {
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
          for (int j = 0; j < MReg; j++) {
            auto vsb = _mm256_sign_epi8(vb, va[j]);
            auto vabsa = _mm256_sign_epi8(va[j], va[j]);
            iacc[j * NReg + i] = _mm256_dpbusd_avx_epi32(iacc[j * NReg + i], vabsa, vsb);
          }
          b4ptr += 8 * KTILE / 2;
          b1ptr += 8 * KTILE / 8;
        }
      }
    }

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
            bacc[i] = _mm256_dpbusd_avx_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm256_dpbusd_avx_epi32(iacc[i], va, vb);
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
            bacc[i] = _mm256_dpbusd_avx_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm256_dpbusd_avx_epi32(iacc[j * NReg + i], va[j], vb);
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
            bacc[i] = _mm256_dpbusd_avx_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm256_dpbusd_avx_epi32(iacc[i], va, vb);
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
            bacc[i] = _mm256_dpbusd_avx_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm256_dpbusd_avx_epi32(iacc[j * NReg + i], va[j], vb);
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
static inline BTLA_CODE gemv_6bit_s8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
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
    for (int i = 0; i < NReg * MReg; i++) {
      iacc[i] = _mm256_setzero_si256();
    }
    if (B.zpptr) {
      __m256i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < NReg; i++) {
        bzp[i] = load_zp_epi8_broadcast_epi32(bzptr + i * 8, vindex);
        bzp[i] = _mm256_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += KTILE) {
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
          for (int j = 0; j < MReg; j++) {
            auto vsb = _mm256_sign_epi8(vb, va[j]);
            auto vabsa = _mm256_sign_epi8(va[j], va[j]);
            iacc[j * NReg + i] = _mm256_dpbusd_avx_epi32(iacc[j * NReg + i], vabsa, vsb);
          }
          b4ptr += 8 * KTILE / 2;
          b2ptr += 8 * KTILE / 4;
        }
      }
    } else {
      for (int ik = 0; ik < blocksize; ik += KTILE) {
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
          for (int j = 0; j < MReg; j++) {
            auto vsb = _mm256_sign_epi8(vb, va[j]);
            auto vabsa = _mm256_sign_epi8(va[j], va[j]);
            iacc[j * NReg + i] = _mm256_dpbusd_avx_epi32(iacc[j * NReg + i], vabsa, vsb);
          }
          b4ptr += 8 * KTILE / 2;
          b2ptr += 8 * KTILE / 4;
        }
      }
    }

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
            bacc[i] = _mm256_dpbusd_avx_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm256_dpbusd_avx_epi32(iacc[i], va, vb);
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
            bacc[i] = _mm256_dpbusd_avx_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm256_dpbusd_avx_epi32(iacc[j * NReg + i], va[j], vb);
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
            bacc[i] = _mm256_dpbusd_avx_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm256_dpbusd_avx_epi32(iacc[i], va, vb);
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
            bacc[i] = _mm256_dpbusd_avx_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm256_dpbusd_avx_epi32(iacc[j * NReg + i], va[j], vb);
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
static inline BTLA_CODE gemv_7bit_s8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
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

  int constexpr KTILE = 4;
  for (int ib = 0; ib < blks; ib += 1) {
    __m256i iacc[NReg * MReg];
    for (int i = 0; i < NReg * MReg; i++) {
      iacc[i] = _mm256_setzero_si256();
    }
    if (B.zpptr) {
      __m256i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < NReg; i++) {
        bzp[i] = load_zp_epi8_broadcast_epi32(bzptr + i * 8, vindex);
        bzp[i] = _mm256_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += KTILE) {
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
          for (int j = 0; j < MReg; j++) {
            auto vsb = _mm256_sign_epi8(vb, va[j]);
            auto vabsa = _mm256_sign_epi8(va[j], va[j]);
            iacc[j * NReg + i] = _mm256_dpbusd_avx_epi32(iacc[j * NReg + i], vabsa, vsb);
          }
          b4ptr += 8 * KTILE / 2;
          b2ptr += 8 * KTILE / 4;
          b1ptr += 8 * KTILE / 8;
        }
      }
    } else {
      for (int ik = 0; ik < blocksize; ik += KTILE) {
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
          for (int j = 0; j < MReg; j++) {
            auto vsb = _mm256_sign_epi8(vb, va[j]);
            auto vabsa = _mm256_sign_epi8(va[j], va[j]);
            iacc[j * NReg + i] = _mm256_dpbusd_avx_epi32(iacc[j * NReg + i], vabsa, vsb);
          }
          b4ptr += 8 * KTILE / 2;
          b2ptr += 8 * KTILE / 4;
          b1ptr += 8 * KTILE / 8;
        }
      }
    }

    gemv_dequant_s32fp32<ScaleT, NReg, MTILE>(A.sptr + ib, A.ldzp, B.sptr + ib * B.ldzp, iacc, acc);
  }

  for (int j = 0; j < MReg; j++) {
    for (int i = 0; i < NReg; i++) {
      _mm256_storeu_ps(C + i * 8 + j * ldc, acc[j * NReg + i]);
    }
  }
  return BTLA_CODE::Success;
}
}  // namespace vnni

#ifdef __GNUC__
#pragma GCC pop_options
#else
#endif
#endif

}  // namespace avx2
}  // namespace kernel
}  // namespace bestla
