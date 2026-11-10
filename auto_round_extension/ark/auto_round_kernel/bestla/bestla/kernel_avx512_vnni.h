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
#if CompileAVX512VNNI()
#if defined(__INTEL_LLVM_COMPILER)
#pragma clang attribute push(__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512bw", "avx512vl", "avx512dq", "avx512vnni")
#endif

namespace vnni {

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
            bacc[i] = _mm512_dpbusd_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm512_dpbusd_epi32(iacc[i], va, vb);
          }
        } else {
          __m512i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm512_set1_epi32(*(int*)(a8ptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_4bits((void*)(b4ptr + i * 32 + (ib * blocksize + ik) * NTILE / 2), vmask);
            vb = _mm512_sub_epi8(vb, bzp[i]);
            bacc[i] = _mm512_dpbusd_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm512_dpbusd_epi32(iacc[j * NReg + i], va[j], vb);
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
            bacc[i] = _mm512_dpbusd_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm512_dpbusd_epi32(iacc[i], va, vb);
          }
        } else {
          __m512i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm512_set1_epi32(*(int*)(a8ptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_4bits((void*)(b4ptr + i * 32 + (ib * blocksize + ik) * NTILE / 2), vmask);
            vb = _mm512_sub_epi8(vb, vbias);
            bacc[i] = _mm512_dpbusd_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm512_dpbusd_epi32(iacc[j * NReg + i], va[j], vb);
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
static inline BTLA_CODE gemv_4bit_s8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  auto& a8ptr = A.aptr;
  auto& b4ptr = B.b4ptr;
  auto& asptr = A.sptr;

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
  const auto vindex = _mm512_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12,
                                      12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0);
  for (int ib = 0; ib < blks; ib += 1) {
    __m512i iacc[NReg * MReg];
    for (int i = 0; i < NReg * MReg; i++) {
      iacc[i] = _mm512_setzero_si512();
    }
    if (B.zpptr) {
      __m512i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < NReg; i++) {
        bzp[i] = load_zp_epi8_broadcast_epi32(bzptr + i * VLen, vindex);
        bzp[i] = _mm512_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += 4) {
        __m512i va[MReg];
        for (int i = 0; i < MReg; i++) {
          va[i] = _mm512_set1_epi32(*(int*)(a8ptr + ib * blocksize + ik + i * A.lda));
        }
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_4bits((void*)(b4ptr + i * 32 + (ib * blocksize + ik) * NTILE / 2), vmask);
          vb = _mm512_sub_epi8(vb, bzp[i]);
          for (int j = 0; j < MReg; j++) {
            auto vsb = _mm512_sign_epi8(vb, va[j]);
            auto vabsa = _mm512_sign_epi8(va[j], va[j]);
            iacc[j * NReg + i] = _mm512_dpbusd_epi32(iacc[j * NReg + i], vabsa, vsb);
          }
        }
      }
    } else {
      for (int ik = 0; ik < blocksize; ik += 4) {
        __m512i va[MReg];
        for (int i = 0; i < MReg; i++) {
          va[i] = _mm512_set1_epi32(*(int*)(a8ptr + ib * blocksize + ik + i * A.lda));
        }
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_4bits((void*)(b4ptr + i * 32 + (ib * blocksize + ik) * NTILE / 2), vmask);
          vb = _mm512_sub_epi8(vb, vbias);
          for (int j = 0; j < MReg; j++) {
            auto vsb = _mm512_sign_epi8(vb, va[j]);
            auto vabsa = _mm512_sign_epi8(va[j], va[j]);
            iacc[j * NReg + i] = _mm512_dpbusd_epi32(iacc[j * NReg + i], vabsa, vsb);
          }
        }
      }
    }

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
            bacc[i] = _mm512_dpbusd_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm512_dpbusd_epi32(iacc[i], va, vb);
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
            bacc[i] = _mm512_dpbusd_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm512_dpbusd_epi32(iacc[j * NReg + i], va[j], vb);
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
            bacc[i] = _mm512_dpbusd_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm512_dpbusd_epi32(iacc[i], va, vb);
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
            bacc[i] = _mm512_dpbusd_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm512_dpbusd_epi32(iacc[j * NReg + i], va[j], vb);
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
static inline BTLA_CODE gemv_2bit_s8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
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
    for (int i = 0; i < NReg * MReg; i++) {
      iacc[i] = _mm512_setzero_si512();
    }

    if (B.zpptr) {
      __m512i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < NReg; i++) {
        bzp[i] = load_zp_epi8_broadcast_epi32(bzptr + i * 16, vindex);
        bzp[i] = _mm512_add_epi8(vbias, bzp[i]);
      }
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        __m512i va[MReg];
        for (int i = 0; i < MReg; i++) {
          va[i] = _mm512_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
        }
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
          vb = _mm512_sub_epi8(vb, bzp[i]);
          for (int j = 0; j < MReg; j++) {
            auto vsb = _mm512_sign_epi8(vb, va[j]);
            auto vabsa = _mm512_sign_epi8(va[j], va[j]);
            iacc[j * NReg + i] = _mm512_dpbusd_epi32(iacc[j * NReg + i], vabsa, vsb);
          }
          b2ptr += VLen * KTILE / 4;
        }
      }
    } else {
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        __m512i va[MReg];
        for (int i = 0; i < MReg; i++) {
          va[i] = _mm512_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
        }
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
          vb = _mm512_sub_epi8(vb, vbias);
          for (int j = 0; j < MReg; j++) {
            auto vsb = _mm512_sign_epi8(vb, va[j]);
            auto vabsa = _mm512_sign_epi8(va[j], va[j]);
            iacc[j * NReg + i] = _mm512_dpbusd_epi32(iacc[j * NReg + i], vabsa, vsb);
          }
          b2ptr += VLen * KTILE / 4;
        }
      }
    }
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
            bacc[i] = _mm512_dpbusd_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm512_dpbusd_epi32(iacc[i], va, vb);
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
            bacc[i] = _mm512_dpbusd_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm512_dpbusd_epi32(iacc[j * NReg + i], va[j], vb);
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
            bacc[i] = _mm512_dpbusd_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm512_dpbusd_epi32(iacc[i], va, vb);
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
            bacc[i] = _mm512_dpbusd_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm512_dpbusd_epi32(iacc[j * NReg + i], va[j], vb);
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
static inline BTLA_CODE gemv_3bit_s8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
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
  int constexpr KTILE = 4;
  for (int ib = 0; ib < blks; ib += 1) {
    __m512i iacc[NReg * MReg];
    for (int i = 0; i < NReg * MReg; i++) {
      iacc[i] = _mm512_setzero_si512();
    }
    if (B.zpptr) {
      __m512i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < NReg; i++) {
        bzp[i] = load_zp_epi8_broadcast_epi32(bzptr + i * 16, vindex);
        bzp[i] = _mm512_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        __m512i va[MReg];
        for (int i = 0; i < MReg; i++) {
          va[i] = _mm512_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
        }
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
          auto vb1 = unpack_1bits(b1ptr, zmm_0x00, zmm_0x04);
          vb = _mm512_or_si512(vb, vb1);
          vb = _mm512_sub_epi8(vb, bzp[i]);
          for (int j = 0; j < MReg; j++) {
            auto vsb = _mm512_sign_epi8(vb, va[j]);
            auto vabsa = _mm512_sign_epi8(va[j], va[j]);
            iacc[j * NReg + i] = _mm512_dpbusd_epi32(iacc[j * NReg + i], vabsa, vsb);
          }
          b2ptr += VLen * KTILE / 4;
          b1ptr += VLen * KTILE / 8;
        }
      }
    } else {
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        __m512i va[MReg];
        for (int i = 0; i < MReg; i++) {
          va[i] = _mm512_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
        }
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
          auto vb1 = unpack_1bits(b1ptr, zmm_0x00, zmm_0x04);
          vb = _mm512_or_si512(vb, vb1);
          vb = _mm512_sub_epi8(vb, vbias);
          for (int j = 0; j < MReg; j++) {
            auto vsb = _mm512_sign_epi8(vb, va[j]);
            auto vabsa = _mm512_sign_epi8(va[j], va[j]);
            iacc[j * NReg + i] = _mm512_dpbusd_epi32(iacc[j * NReg + i], vabsa, vsb);
          }
          b2ptr += VLen * KTILE / 4;
          b1ptr += VLen * KTILE / 8;
        }
      }
    }

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
            bacc[i] = _mm512_dpbusd_epi32(bacc[i], onesu8, vb1);
            iacc[i] = _mm512_dpbusd_epi32(iacc[i], va, vb1);
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
            bacc[i] = _mm512_dpbusd_epi32(bacc[i], onesu8, vb1);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm512_dpbusd_epi32(iacc[j * NReg + i], va[j], vb1);
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
            bacc[i] = _mm512_dpbusd_epi32(bacc[i], onesu8, vb1);
            iacc[i] = _mm512_dpbusd_epi32(iacc[i], va, vb1);
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
            bacc[i] = _mm512_dpbusd_epi32(bacc[i], onesu8, vb1);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm512_dpbusd_epi32(iacc[j * NReg + i], va[j], vb1);
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
static inline BTLA_CODE gemv_1bit_s8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
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
  int constexpr KTILE = 4;
  for (int ib = 0; ib < blks; ib += 1) {
    __m512i iacc[NReg * MReg];
    for (int i = 0; i < NReg * MReg; i++) {
      iacc[i] = _mm512_setzero_si512();
    }
    if (B.zpptr) {
      __m512i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < NReg; i++) {
        bzp[i] = load_zp_epi8_broadcast_epi32(bzptr + i * 16, vindex);
        bzp[i] = _mm512_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        __m512i va[MReg];
        for (int i = 0; i < MReg; i++) {
          va[i] = _mm512_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
        }
        for (int i = 0; i < NReg; i++) {
          auto vb1 = unpack_1bits(b1ptr, zmm_0x00, zmm_0x04);
          vb1 = _mm512_srli_epi32(vb1, 2);
          vb1 = _mm512_sub_epi8(vb1, bzp[i]);
          for (int j = 0; j < MReg; j++) {
            auto vsb = _mm512_sign_epi8(vb1, va[j]);
            auto vabsa = _mm512_sign_epi8(va[j], va[j]);
            iacc[j * NReg + i] = _mm512_dpbusd_epi32(iacc[j * NReg + i], vabsa, vsb);
          }
          b1ptr += VLen * KTILE / 8;
        }
      }
    } else {
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        __m512i va[MReg];
        for (int i = 0; i < MReg; i++) {
          va[i] = _mm512_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
        }
        for (int i = 0; i < NReg; i++) {
          auto vb1 = unpack_1bits(b1ptr, zmm_0x00, zmm_0x04);
          vb1 = _mm512_srli_epi32(vb1, 2);
          vb1 = _mm512_sub_epi8(vb1, vbias);
          for (int j = 0; j < MReg; j++) {
            auto vsb = _mm512_sign_epi8(vb1, va[j]);
            auto vabsa = _mm512_sign_epi8(va[j], va[j]);
            iacc[j * NReg + i] = _mm512_dpbusd_epi32(iacc[j * NReg + i], vabsa, vsb);
          }
          b1ptr += VLen * KTILE / 8;
        }
      }
    }

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
            bacc[i] = _mm512_dpbusd_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm512_dpbusd_epi32(iacc[i], va, vb);
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
            bacc[i] = _mm512_dpbusd_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm512_dpbusd_epi32(iacc[j * NReg + i], va[j], vb);
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
            bacc[i] = _mm512_dpbusd_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm512_dpbusd_epi32(iacc[i], va, vb);
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
            bacc[i] = _mm512_dpbusd_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm512_dpbusd_epi32(iacc[j * NReg + i], va[j], vb);
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
static inline BTLA_CODE gemv_5bit_s8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
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
  int constexpr KTILE = 4;
  for (int ib = 0; ib < blks; ib += 1) {
    __m512i iacc[NReg * MReg];
    for (int i = 0; i < NReg * MReg; i++) {
      iacc[i] = _mm512_setzero_si512();
    }
    if (B.zpptr) {
      __m512i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < NReg; i++) {
        bzp[i] = load_zp_epi8_broadcast_epi32(bzptr + i * 16, vindex);
        bzp[i] = _mm512_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += KTILE) {
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
          for (int j = 0; j < MReg; j++) {
            auto vsb = _mm512_sign_epi8(vb, va[j]);
            auto vabsa = _mm512_sign_epi8(va[j], va[j]);
            iacc[j * NReg + i] = _mm512_dpbusd_epi32(iacc[j * NReg + i], vabsa, vsb);
          }
          b4ptr += VLen * KTILE / 2;
          b1ptr += VLen * KTILE / 8;
        }
      }
    } else {
      for (int ik = 0; ik < blocksize; ik += KTILE) {
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
          for (int j = 0; j < MReg; j++) {
            auto vsb = _mm512_sign_epi8(vb, va[j]);
            auto vabsa = _mm512_sign_epi8(va[j], va[j]);
            iacc[j * NReg + i] = _mm512_dpbusd_epi32(iacc[j * NReg + i], vabsa, vsb);
          }
          b4ptr += VLen * KTILE / 2;
          b1ptr += VLen * KTILE / 8;
        }
      }
    }

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
            bacc[i] = _mm512_dpbusd_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm512_dpbusd_epi32(iacc[i], va, vb);
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
            bacc[i] = _mm512_dpbusd_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm512_dpbusd_epi32(iacc[j * NReg + i], va[j], vb);
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
            bacc[i] = _mm512_dpbusd_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm512_dpbusd_epi32(iacc[i], va, vb);
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
            bacc[i] = _mm512_dpbusd_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm512_dpbusd_epi32(iacc[j * NReg + i], va[j], vb);
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
static inline BTLA_CODE gemv_6bit_s8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
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
  int constexpr KTILE = 4;
  for (int ib = 0; ib < blks; ib += 1) {
    __m512i iacc[NReg * MReg];
    for (int i = 0; i < NReg * MReg; i++) {
      iacc[i] = _mm512_setzero_si512();
    }
    if (B.zpptr) {
      __m512i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < NReg; i++) {
        bzp[i] = load_zp_epi8_broadcast_epi32(bzptr + i * 16, vindex);
        bzp[i] = _mm512_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += KTILE) {
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
          for (int j = 0; j < MReg; j++) {
            auto vsb = _mm512_sign_epi8(vb, va[j]);
            auto vabsa = _mm512_sign_epi8(va[j], va[j]);
            iacc[j * NReg + i] = _mm512_dpbusd_epi32(iacc[j * NReg + i], vabsa, vsb);
          }
          b4ptr += VLen * KTILE / 2;
          b2ptr += VLen * KTILE / 4;
        }
      }
    } else {
      for (int ik = 0; ik < blocksize; ik += KTILE) {
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
          for (int j = 0; j < MReg; j++) {
            auto vsb = _mm512_sign_epi8(vb, va[j]);
            auto vabsa = _mm512_sign_epi8(va[j], va[j]);
            iacc[j * NReg + i] = _mm512_dpbusd_epi32(iacc[j * NReg + i], vabsa, vsb);
          }
          b4ptr += VLen * KTILE / 2;
          b2ptr += VLen * KTILE / 4;
        }
      }
    }

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
            bacc[i] = _mm512_dpbusd_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm512_dpbusd_epi32(iacc[i], va, vb);
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
            bacc[i] = _mm512_dpbusd_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm512_dpbusd_epi32(iacc[j * NReg + i], va[j], vb);
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
            bacc[i] = _mm512_dpbusd_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm512_dpbusd_epi32(iacc[i], va, vb);
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
            bacc[i] = _mm512_dpbusd_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm512_dpbusd_epi32(iacc[j * NReg + i], va[j], vb);
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

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_7bit_s8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
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
  int constexpr KTILE = 4;
  for (int ib = 0; ib < blks; ib += 1) {
    __m512i iacc[NReg * MReg];
    for (int i = 0; i < NReg * MReg; i++) {
      iacc[i] = _mm512_setzero_si512();
    }
    if (B.zpptr) {
      __m512i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < NReg; i++) {
        bzp[i] = load_zp_epi8_broadcast_epi32(bzptr + i * 16, vindex);
        bzp[i] = _mm512_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += KTILE) {
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
          for (int j = 0; j < MReg; j++) {
            auto vsb = _mm512_sign_epi8(vb, va[j]);
            auto vabsa = _mm512_sign_epi8(va[j], va[j]);
            iacc[j * NReg + i] = _mm512_dpbusd_epi32(iacc[j * NReg + i], vabsa, vsb);
          }
          b4ptr += VLen * KTILE / 2;
          b2ptr += VLen * KTILE / 4;
          b1ptr += VLen * KTILE / 8;
        }
      }
    } else {
      for (int ik = 0; ik < blocksize; ik += KTILE) {
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
          for (int j = 0; j < MReg; j++) {
            auto vsb = _mm512_sign_epi8(vb, va[j]);
            auto vabsa = _mm512_sign_epi8(va[j], va[j]);
            iacc[j * NReg + i] = _mm512_dpbusd_epi32(iacc[j * NReg + i], vabsa, vsb);
          }
          b4ptr += VLen * KTILE / 2;
          b2ptr += VLen * KTILE / 4;
          b1ptr += VLen * KTILE / 8;
        }
      }
    }

    gemv_dequant_s32fp32<ScaleT, NReg, MTILE>(A.sptr + ib, A.ldzp, B.sptr + ib * B.ldzp, iacc, acc);
  }

  for (int j = 0; j < MReg; j++) {
    for (int i = 0; i < NReg; i++) {
      _mm512_storeu_ps(C + i * VLen + j * ldc, acc[j * NReg + i]);
    }
  }
  return BTLA_CODE::Success;
}

}  // namespace vnni

#if defined(__INTEL_LLVM_COMPILER)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif
}  // namespace avx512f
}  // namespace kernel
}  // namespace bestla
