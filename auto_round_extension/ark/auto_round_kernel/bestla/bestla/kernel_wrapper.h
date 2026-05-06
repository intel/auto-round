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
#include <array>
#include <cassert>
#include <type_traits>
#include <immintrin.h>

#include "bestla.h"
#include "bestla_utils.h"
#include "kernel_ref.h"
#include "kernel_jit.h"
#include "kernel_avx2.h"
#include "kernel_avx_vnni.h"
#include "kernel_avx512f.h"
#include "kernel_avx512_vnni.h"
#include "kernel_avx512_bf16.h"
#include "kernel_avx512_fp16.h"

namespace bestla {
namespace kernel {
namespace wrapper {

class ZeroReg {
 public:
  static void forward() {
#if CompileAVX2()
    kernel::avx2::zero_reg();
#endif
  }
};

template <int NTILE, int RowPack, typename T_SRC, typename T_DST = T_SRC>
class PaddingInterleaveMN {
  // M x N ===> N/NTile x M/RowPack x NTile x RowPack (leading dim stride = NTile * dststride)
 public:
  TLACALL BTLA_CODE forward(const T_SRC* src, T_DST* dst, int row, int col, int row_pad, int col_pad, int src_step,
                            int dst_step) {
#if CompileFP16()
    if constexpr (utils::isa_base<ISA_T>::avx512_fp16 && NTILE % 16 == 0) {
      const auto kern_ret = kernel::avx512f::avx512_fp16::padding_interleave_cvt<T_SRC, T_DST, RowPack>::forward(
          src, dst, NTILE, row, col, row_pad, col_pad, src_step, dst_step);
      if (kern_ret != BTLA_CODE::NotSupport) return kern_ret;
    }
#endif
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f && NTILE % 16 == 0) {
      const auto kern_ret = kernel::avx512f::padding_interleave_cvt<T_SRC, T_DST, RowPack>::forward(
          src, dst, NTILE, row, col, row_pad, col_pad, src_step, dst_step);
      if (kern_ret != BTLA_CODE::NotSupport) return kern_ret;
    }
#endif
    return ref::padding_interleave(src, dst, row, col, row_pad, col_pad, src_step, dst_step, NTILE, RowPack);
  }

  AUTOCALL BTLA_CODE forward_auto(const T_SRC* src, T_DST* dst, int row, int col, int row_pad, int col_pad,
                                  int src_step, int dst_step) {
    GetCPUDevice();
    if (_cd->AVX512_FP16()) {
      return forward<BTLA_ISA::AVX512_FP16>(src, dst, row, col, row_pad, col_pad, src_step, dst_step);
    }
    if (_cd->AVX512F()) {
      return forward<BTLA_ISA::AVX512F>(src, dst, row, col, row_pad, col_pad, src_step, dst_step);
    }
    if (_cd->AVX2()) {
      return forward<BTLA_ISA::AVX2>(src, dst, row, col, row_pad, col_pad, src_step, dst_step);
    }
    return forward<BTLA_ISA::NoSIMD>(src, dst, row, col, row_pad, col_pad, src_step, dst_step);
  }
};

template <int NTile, int RowPack, typename T_SRC, typename T_DST = T_SRC>
class RevertPaddingInterleaveMN {
  // M x N ===> N/NTile x M/RowPack x NTile x RowPack (leading dim stride = NTile * dststride)
 public:
  TLACALL BTLA_CODE forward(const T_SRC* src, T_DST* dst, int row, int col, int row_pad, int col_pad, int src_step,
                            int dst_step) {
    return ref::revert_padding_interleave(src, dst, row, col, row_pad, col_pad, src_step, dst_step, NTile, RowPack);
  }

  AUTOCALL BTLA_CODE forward_auto(const T_SRC* src, T_DST* dst, int row, int col, int row_pad, int col_pad,
                                  int src_step, int dst_step) {
    return forward<BTLA_ISA::NoSIMD>(src, dst, row, col, row_pad, col_pad, src_step, dst_step);
  }
};

template <int MTile, int ColPack, typename T_SRC, typename T_DST = T_SRC>
class PaddingTransInterleaveMN {
  // row and cols are in terms of src
  // M x N ===> M/MTile x N/ColPack x MTile x ColPack (leading dim stride = MTile * dststride)
 public:
  TLACALL BTLA_CODE forward(const T_SRC* src, T_DST* dst, int row, int col, int row_pad, int col_pad, int src_step,
                            int dst_step) {
#if CompileFP16()
    if constexpr (utils::isa_base<ISA_T>::avx512_fp16) {
      const auto kern_ret = kernel::avx512f::avx512_fp16::padding_trans_interleave_cvt<T_SRC, T_DST, ColPack>::forward(
          src, dst, MTile, row, col, row_pad, col_pad, src_step, dst_step);
      if (kern_ret != BTLA_CODE::NotSupport) return kern_ret;
    }
#endif
#if CompileAVX512F()
    // Note: rows/cols and i/j are in terms of src
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      const auto kern_ret = kernel::avx512f::padding_trans_interleave_cvt<T_SRC, T_DST, ColPack>::forward(
          src, dst, MTile, row, col, row_pad, col_pad, src_step, dst_step);
      if (kern_ret != BTLA_CODE::NotSupport) return kern_ret;
    }
#endif
    return ref::padding_trans_interleave(src, dst, row, col, row_pad, col_pad, src_step, dst_step, MTile, ColPack);
  }

  AUTOCALL BTLA_CODE forward_auto(const T_SRC* src, T_DST* dst, int row, int col, int row_pad, int col_pad,
                                  int src_step, int dst_step) {
    GetCPUDevice();
    if (_cd->AVX512_FP16()) {
      return forward<BTLA_ISA::AVX512_FP16>(src, dst, row, col, row_pad, col_pad, src_step, dst_step);
    }
    if (_cd->AVX512F()) {
      return forward<BTLA_ISA::AVX512F>(src, dst, row, col, row_pad, col_pad, src_step, dst_step);
    }
    return forward<BTLA_ISA::NoSIMD>(src, dst, row, col, row_pad, col_pad, src_step, dst_step);
  }
};

class Memcpy2D {
 public:
  template <BTLA_ISA ISA_T, typename _SRC_T, typename _DST_T>
  static BTLA_CODE forward(const _SRC_T* srcptr, _DST_T* dstptr, int row, int col, int srcstep, int dststep,
                           void* const_elt_v = nullptr) {
    auto ret = BTLA_CODE::NotSupport;
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      ret = kernel::jit::JitMemcpy2DAvx512f::forward<_SRC_T, _DST_T>(srcptr, dstptr, row, col, srcstep, dststep,
                                                                     const_elt_v);
      if (ret == BTLA_CODE::Success) {
        return ret;
      }
    }
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      auto align_col = static_cast<int>(col * sizeof(_SRC_T) / 32 * 32 / sizeof(_SRC_T));
      ret = kernel::jit::JitMemcpy2DAvx2::forward<_SRC_T, _DST_T>(srcptr, dstptr, row, align_col, srcstep, dststep,
                                                                  const_elt_v);
      if (col - align_col > 0)
        ret = kernel::ref::memcpy2d(
            srcptr + align_col, dstptr + align_col, row, static_cast<int>((col - align_col) * sizeof(_SRC_T)),
            static_cast<int>(srcstep * sizeof(_SRC_T)), static_cast<int>(dststep * sizeof(_DST_T)));
      if (ret == BTLA_CODE::Success) {
        return ret;
      }
    }
    return kernel::ref::memcpy2d(srcptr, dstptr, row, static_cast<int>(col * sizeof(_SRC_T)),
                                 static_cast<int>(srcstep * sizeof(_SRC_T)),
                                 static_cast<int>(dststep * sizeof(_DST_T)));
  }

  template <BTLA_ISA ISA_T, typename _SRC_T, typename _DST_T, BTLA_ELTWISEOP OP_T>
  static BTLA_CODE forward1(const _SRC_T* srcptr, _DST_T* dstptr, int row, int col, int srcstep, int dststep,
                            void* const_elt_v = nullptr) {
    auto ret = BTLA_CODE::NotSupport;
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      ret = kernel::jit::JitMemcpy2DAvx512f::forward1<_SRC_T, _DST_T, OP_T>(srcptr, dstptr, row, col, srcstep, dststep,
                                                                            const_elt_v);
      if (ret == BTLA_CODE::Success) {
        return ret;
      }
    }
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      auto align_col = static_cast<int>(col * sizeof(_SRC_T) / 32 * 32 / sizeof(_SRC_T));
      ret = kernel::jit::JitMemcpy2DAvx2::forward1<_SRC_T, _DST_T, OP_T>(srcptr, dstptr, row, align_col, srcstep,
                                                                         dststep, const_elt_v);
      if (col - align_col > 0)
        ret = kernel::ref::memcpy2d_withop<_SRC_T, _DST_T, OP_T>(srcptr + align_col, dstptr + align_col, row,
                                                                 col - align_col, srcstep, dststep, const_elt_v);
      if (ret == BTLA_CODE::Success) {
        return ret;
      }
    }
    return ref::memcpy2d_withop<_SRC_T, _DST_T, OP_T>(srcptr, dstptr, row, col, srcstep, dststep, const_elt_v);
  }

  template <typename _SRC_T, typename _DST_T, BTLA_ELTWISEOP OP_T>
  static BTLA_CODE forward1_auto(const _SRC_T* srcptr, _DST_T* dstptr, int row, int col, int srcstep, int dststep,
                                 void* const_elt_v = nullptr) {
    GetCPUDevice();
    if (_cd->AVX512F()) {
      return forward1<BTLA_ISA::AVX512F, _SRC_T, _DST_T, OP_T>(srcptr, dstptr, row, col, srcstep, dststep, const_elt_v);
    }
    if (_cd->AVX2()) {
      return forward1<BTLA_ISA::AVX2, _SRC_T, _DST_T, OP_T>(srcptr, dstptr, row, col, srcstep, dststep, const_elt_v);
    }
    return forward1<BTLA_ISA::NoSIMD, _SRC_T, _DST_T, OP_T>(srcptr, dstptr, row, col, srcstep, dststep, const_elt_v);
  }

  template <typename _SRC_T, typename _DST_T>
  static BTLA_CODE forward_auto(const _SRC_T* srcptr, _DST_T* dstptr, int row, int col, int srcstep, int dststep) {
    GetCPUDevice();
    if (_cd->AVX512F()) {
      return forward<BTLA_ISA::AVX512F, _SRC_T, _DST_T>(srcptr, dstptr, row, col, srcstep, dststep);
    }
    if (_cd->AVX2()) {
      return forward<BTLA_ISA::AVX2, _SRC_T, _DST_T>(srcptr, dstptr, row, col, srcstep, dststep);
    }
    return forward<BTLA_ISA::NoSIMD, _SRC_T, _DST_T>(srcptr, dstptr, row, col, srcstep, dststep);
  }
};

class Memcpy2DFp32CvtBf16 {
 public:
  template <BTLA_ISA ISA_T>
  static BTLA_CODE forward(const void* srcptr, void* dstptr, int row, int col, int srcstride, int dststride,
                           bool zeropadding) {
#if CompileBF16()
    if constexpr (utils::isa_base<ISA_T>::avx512_bf16) {
      return kernel::avx512f::avx512_bf16::fp32_cvt_bf16_2D_write_back(srcptr, dstptr, row, col, srcstride, dststride,
                                                                       zeropadding);
    }
#endif
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return kernel::avx512f::cvt_fp32_T_2D((const float*)srcptr, (utils::bf16*)dstptr, row, col,
                                            srcstride / sizeof(float), dststride / sizeof(utils::bf16), zeropadding);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return kernel::avx2::cvt_fp32_T_2D((const float*)srcptr, (utils::bf16*)dstptr, row, col,
                                         srcstride / sizeof(float), dststride / sizeof(utils::bf16), zeropadding);
    }
#endif
    return kernel::ref::dt_cvt_2D_write_back<float, utils::bf16>(srcptr, dstptr, row, col, srcstride, dststride,
                                                                 zeropadding);
  }
};

class Memcpy2DFp32CvtFp16 {
 public:
  template <BTLA_ISA ISA_T>
  static BTLA_CODE forward(const void* srcptr, void* dstptr, int row, int col, int srcstride, int dststride,
                           bool zeropadding) {
#if CompileFP16()
    if constexpr (utils::isa_base<ISA_T>::avx512_fp16) {
      return kernel::avx512f::avx512_fp16::fp32_cvt_fp16_2D_write_back(
          reinterpret_cast<const float*>(srcptr), reinterpret_cast<utils::fp16*>(dstptr), row, col,
          srcstride / sizeof(float), dststride / sizeof(utils::fp16), zeropadding);
    }
#endif
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return kernel::avx512f::cvt_fp32_T_2D(reinterpret_cast<const float*>(srcptr),
                                            reinterpret_cast<utils::fp16*>(dstptr), row, col, srcstride / sizeof(float),
                                            dststride / sizeof(utils::fp16), zeropadding);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return kernel::avx2::cvt_fp32_T_2D(reinterpret_cast<const float*>(srcptr), reinterpret_cast<utils::fp16*>(dstptr),
                                         row, col, srcstride / sizeof(float), dststride / sizeof(utils::fp16),
                                         zeropadding);
    }
#endif
    return kernel::ref::dt_cvt_2D_write_back<float, utils::fp16>(srcptr, dstptr, row, col, srcstride, dststride,
                                                                 zeropadding);
  }
};

template <typename T>
class Memcpy2DFp32TPadding {
 public:
  TLACALL BTLA_CODE forward(const float* srcptr, T* dstptr, int row, int col, int srcstride, int dststride,
                            bool zeropadding) {
    if constexpr (std::is_same_v<T, utils::fp16>) {
      return Memcpy2DFp32CvtFp16::forward<ISA_T>(srcptr, dstptr, row, col, srcstride, dststride, zeropadding);
    }
    if constexpr (std::is_same_v<T, utils::bf16>) {
      return Memcpy2DFp32CvtBf16::forward<ISA_T>(srcptr, dstptr, row, col, srcstride, dststride, zeropadding);
    }
    return Memcpy2D::forward<ISA_T, float, T>(srcptr, dstptr, row, col, srcstride / sizeof(float),
                                              dststride / sizeof(T), nullptr);
  }

  AUTOCALL BTLA_CODE forward_auto(const float* srcptr, T* dstptr, int row, int col, int srcstride, int dststride,
                                  bool zeropadding) {
    GetCPUDevice();
    if (_cd->AVX512_FP16()) {
      return forward<BTLA_ISA::AVX512_FP16>(srcptr, dstptr, row, col, srcstride, dststride, zeropadding);
    }
    if (_cd->AVX512_BF16()) {
      return forward<BTLA_ISA::AVX512_BF16>(srcptr, dstptr, row, col, srcstride, dststride, zeropadding);
    }
    if (_cd->AVX512F()) {
      return forward<BTLA_ISA::AVX512F>(srcptr, dstptr, row, col, srcstride, dststride, zeropadding);
    }
    if (_cd->AVX2()) {
      return forward<BTLA_ISA::AVX2>(srcptr, dstptr, row, col, srcstride, dststride, zeropadding);
    }
    return forward<BTLA_ISA::NoSIMD>(srcptr, dstptr, row, col, srcstride, dststride, zeropadding);
  }
};

class Memcpy2DFp16CvtFp32 {
 public:
  template <BTLA_ISA ISA_T>
  static BTLA_CODE forward(const void* srcptr, void* dstptr, int row, int col, int srcstride, int dststride,
                           bool zeropadding) {
#if CompileFP16()
    if constexpr (utils::isa_base<ISA_T>::avx512_fp16) {
      return kernel::avx512f::avx512_fp16::fp16_cvt_fp32_2D_write_back(  //
          reinterpret_cast<const utils::fp16*>(srcptr), reinterpret_cast<float*>(dstptr), row, col,
          srcstride / sizeof(utils::fp16), dststride / sizeof(float), zeropadding);
    }
#endif
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return kernel::avx512f::cvt_T_fp32_2D(  //
          reinterpret_cast<const utils::fp16*>(srcptr), reinterpret_cast<float*>(dstptr), row, col,
          srcstride / sizeof(utils::fp16), dststride / sizeof(float), zeropadding);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return kernel::avx2::cvt_T_fp32_2D(  //
          reinterpret_cast<const utils::fp16*>(srcptr), reinterpret_cast<float*>(dstptr), row, col,
          srcstride / sizeof(utils::fp16), dststride / sizeof(float), zeropadding);
    }
#endif
    return kernel::ref::dt_cvt_2D_write_back<utils::fp16, float>(srcptr, dstptr, row, col, srcstride, dststride,
                                                                 zeropadding);
  }
};

class Memcpy2DBf16CvtFp32 {
 public:
  template <BTLA_ISA ISA_T>
  static BTLA_CODE forward(const void* srcptr, void* dstptr, int row, int col, int srcstride, int dststride,
                           bool zeropadding) {
#if CompileBF16()
    if constexpr (utils::isa_base<ISA_T>::avx512_bf16) {
      return kernel::avx512f::avx512_bf16::bf16_cvt_fp32_2D_write_back(  //
          reinterpret_cast<const utils::bf16*>(srcptr), reinterpret_cast<float*>(dstptr), row, col,
          srcstride / sizeof(utils::bf16), dststride / sizeof(float), zeropadding);
    }
#endif
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return kernel::avx512f::cvt_T_fp32_2D(reinterpret_cast<const utils::bf16*>(srcptr),
                                            reinterpret_cast<float*>(dstptr), row, col, srcstride / sizeof(utils::bf16),
                                            dststride / sizeof(float), zeropadding);
    }
#endif
#if CompileAVX2()
    if constexpr (ISA_T >= BTLA_ISA::AVX2) {
      return kernel::avx2::cvt_T_fp32_2D(reinterpret_cast<const utils::bf16*>(srcptr), reinterpret_cast<float*>(dstptr),
                                         row, col, srcstride / sizeof(utils::bf16), dststride / sizeof(float),
                                         zeropadding);
    }
#endif
    return kernel::ref::dt_cvt_2D_write_back<utils::bf16, float>(srcptr, dstptr, row, col, srcstride, dststride,
                                                                 zeropadding);
  }
};

class Memcpy2DFp8e4m3CvtFp32 {
 public:
  template <BTLA_ISA ISA_T>
  static BTLA_CODE forward(const void* srcptr, void* dstptr, int row, int col, int srcstride, int dststride,
                           bool zeropadding) {
    return kernel::ref::dt_cvt_2D_write_back<utils::f8e4m3, float>(srcptr, dstptr, row, col, srcstride, dststride,
                                                                   zeropadding);
  }
};

class Memcpy2DPadding {
 public:
  static BTLA_CODE forward(const void* srcptr, void* dstptr, int row, int colsize, int srcstride, int dststride,
                           bool zeropadding) {
    auto srcp = (char*)srcptr;
    if (zeropadding && dststride != colsize) {
      for (int i = 0; i < row; i++) {
        auto dstp = (char*)dstptr + i * dststride;
        std::memcpy(dstp, srcp + i * srcstride, colsize);
        std::memset(dstp + colsize, 0, (dststride - colsize));
      }
    } else {
      for (int i = 0; i < row; i++) {
        auto dstp = (char*)dstptr + i * dststride;
        std::memcpy(dstp, srcp + i * srcstride, colsize);
      }
    }
    return BTLA_CODE::Success;
  }
};

class Memcpy2DCvtPadding {
 public:
  template <BTLA_ISA ISA_T>
  static BTLA_CODE forward(const void* srcptr, BTLA_DTYPE srct, void* dstptr, BTLA_DTYPE dstt, int row, int col,
                           int srcstep, int dststep, bool zeropadding, void* tmpcache, size_t cachesize) {
    bool needf32 = false;
    int rowstep = utils::downdiv(cachesize, col * 4);
    rowstep = std::min(row, rowstep);
    if (dstt != BTLA_DTYPE::F32 && srct != BTLA_DTYPE::F32) {
      needf32 = true;
      assert(col <= cachesize * 4);
    }
    auto srcbytes = utils::bestla_dtype_bytes(srct);
    auto dstbytes = utils::bestla_dtype_bytes(dstt);
    if (srct == dstt) {
      return Memcpy2D::forward_auto<int8_t, int8_t>((const int8_t*)srcptr, (int8_t*)dstptr, row, col * srcbytes,
                                                    srcstep * srcbytes, dststep * dstbytes);
    }
    if (needf32) {
      auto tmpf32 = (float*)tmpcache;
      int cachestride = col * 4;
      for (int irow = 0; irow < row; irow += rowstep) {
        auto row_re = utils::remainsize(irow, row, rowstep);
        if (srct == BTLA_DTYPE::F16) {
          Memcpy2DFp16CvtFp32::forward<ISA_T>((utils::fp16*)srcptr + irow * srcstep, tmpf32, row_re, col, srcstep * 2,
                                              cachestride, false);
        }
        if (srct == BTLA_DTYPE::BF16) {
          Memcpy2DBf16CvtFp32::forward<ISA_T>((utils::bf16*)srcptr + irow * srcstep, tmpf32, row_re, col, srcstep * 2,
                                              cachestride, false);
        }
        if (dstt == BTLA_DTYPE::F16) {
          Memcpy2DFp32CvtFp16::forward<ISA_T>(tmpf32, (utils::fp16*)dstptr + irow * dststep, row_re, col, cachestride,
                                              dststep * 2, zeropadding);
        }
        if (dstt == BTLA_DTYPE::BF16) {
          Memcpy2DFp32CvtBf16::forward<ISA_T>(tmpf32, (utils::bf16*)dstptr + irow * dststep, row_re, col, cachestride,
                                              dststep * 2, zeropadding);
        }
      }
      return BTLA_CODE::Success;
    } else {
      if (dstt == BTLA_DTYPE::F32) {
        if (srct == BTLA_DTYPE::F16) {
          return Memcpy2DFp16CvtFp32::forward<ISA_T>(srcptr, dstptr, row, col, srcstep * 2, dststep * 4, zeropadding);
        }
        if (srct == BTLA_DTYPE::BF16) {
          return Memcpy2DBf16CvtFp32::forward<ISA_T>(srcptr, dstptr, row, col, srcstep * 2, dststep * 4, zeropadding);
        }
        if (srct == BTLA_DTYPE::F8_E4M3) {
          return Memcpy2DFp8e4m3CvtFp32::forward<ISA_T>(srcptr, dstptr, row, col, srcstep, dststep * 4, zeropadding);
        }
      }
      if (dstt == BTLA_DTYPE::BF16) {
        assert(srct == BTLA_DTYPE::F32);
        return Memcpy2DFp32CvtBf16::forward<ISA_T>(srcptr, dstptr, row, col, srcstep * 4, dststep * 2, zeropadding);
      }
      if (dstt == BTLA_DTYPE::F16) {
        assert(srct == BTLA_DTYPE::F32);
        return Memcpy2DFp32CvtFp16::forward<ISA_T>(srcptr, dstptr, row, col, srcstep * 4, dststep * 2, zeropadding);
      }
    }
    return BTLA_CODE::NotSupport;
  }

  static BTLA_CODE forward_auto(const void* srcptr, BTLA_DTYPE srct, void* dstptr, BTLA_DTYPE dstt, int row, int col,
                                int srcstep, int dststep, bool zeropadding, void* tmpcache, size_t cachesize) {
    GetCPUDevice();
    if (_cd->AVX512_FP16()) {
      return forward<BTLA_ISA::AVX512_FP16>(srcptr, srct, dstptr, dstt, row, col, srcstep, dststep, zeropadding,
                                            tmpcache, cachesize);
    }
    if (_cd->AVX512_BF16()) {
      return forward<BTLA_ISA::AVX512_BF16>(srcptr, srct, dstptr, dstt, row, col, srcstep, dststep, zeropadding,
                                            tmpcache, cachesize);
    }
    if (_cd->AVX512F()) {
      return forward<BTLA_ISA::AVX512F>(srcptr, srct, dstptr, dstt, row, col, srcstep, dststep, zeropadding, tmpcache,
                                        cachesize);
    }
    if (_cd->AVX2()) {
      return forward<BTLA_ISA::AVX2>(srcptr, srct, dstptr, dstt, row, col, srcstep, dststep, zeropadding, tmpcache,
                                     cachesize);
    }
    return forward<BTLA_ISA::NoSIMD>(srcptr, srct, dstptr, dstt, row, col, srcstep, dststep, zeropadding, tmpcache,
                                     cachesize);
  }
};
template <typename _DST_T, int _PACK_ROW>
class DecompressDQKBlockS4Fp {
 public:
  template <BTLA_ISA ISA_T, BTLA_DTYPE S4_T>
  static inline BTLA_CODE forward(utils::int4x2* srcptr, _DST_T* dstptr, int row, int col, int ld_src, int ld_dst,
                                  uint8_t* scales, float* dq_scale, int k_offset, int n_offset, int kblock, int NPad,
                                  int N, int dq_blk, int dq_offset_idx, void* tmp, size_t tmpsize) {
    BTLA_CODE ret = BTLA_CODE::NotSupport;
    ret = ref::decompress_dq_kblock_s4_fp<S4_T, _DST_T, _PACK_ROW>(
        srcptr, dstptr, row, col, ld_src, ld_dst, scales, dq_scale, k_offset, n_offset, kblock, dq_blk, dq_offset_idx,
        NPad, N, reinterpret_cast<int8_t*>(tmp), tmpsize);
    return ret;
  }
};

class Dq8GetScale {
 public:
  template <BTLA_ISA ISA_T>
  static BTLA_CODE forward(uint8_t* src, float* dst, int row, int col, int scale_offset, int dq_blk, int dq_offset_idx,
                           float* dq_scale, int src_stride, int dst_stride, bool zeropadding, int mN) {
#if CompileAVX512F()
    if (ISA_T >= BTLA_ISA::AVX512F) {
      return kernel::avx512f::dq8_get_fp_scale(src, dst, row, col, scale_offset, dq_blk, dq_offset_idx, dq_scale,
                                               src_stride, dst_stride, zeropadding, mN);
    }
#endif
#if CompileAVX2()
    if (ISA_T >= BTLA_ISA::AVX2) {
      return kernel::avx2::dq8_get_fp_scale(src, dst, row, col, scale_offset, dq_blk, dq_offset_idx, dq_scale,
                                            src_stride, dst_stride, zeropadding, mN);
    }
#endif
    return kernel::ref::dq8_get_fp_scale(src, dst, row, col, scale_offset, dq_blk, dq_offset_idx, dq_scale, src_stride,
                                         dst_stride, zeropadding, mN);
  }
};

class CompressS8S4 {
 public:
  TLACALL BTLA_CODE forward(const int8_t* srcptr, bestla::utils::int4x2* dstptr, size_t size) {
    return ref::compress_s8_s4(srcptr, dstptr, size);
  }

  AUTOCALL BTLA_CODE forward_auto(const int8_t* srcptr, bestla::utils::int4x2* dstptr, size_t size) {
    return forward<BTLA_ISA::NoSIMD>(srcptr, dstptr, size);
  }
};

class CompressFp4 {
 public:
  TLACALL BTLA_CODE forward(const int8_t* srcptr, bestla::utils::f4x2* dstptr, size_t size) {
    return ref::compress_f4(srcptr, dstptr, size);
  }

  AUTOCALL BTLA_CODE forward_auto(const int8_t* srcptr, bestla::utils::f4x2* dstptr, size_t size) {
    return forward<BTLA_ISA::NoSIMD>(srcptr, dstptr, size);
  }
};

class CompressBit7 {
 public:
  TLACALL BTLA_CODE forward(const int8_t* srcptr, bestla::utils::bit4x2* bit4ptr, utils::bit2x4* bit2ptr,
                            utils::bit1x8* bit1ptr, size_t size) {
    return ref::compress_7bit(srcptr, bit4ptr, bit2ptr, bit1ptr, size);
  }

  AUTOCALL BTLA_CODE forward_auto(const int8_t* srcptr, bestla::utils::bit4x2* bit4ptr, utils::bit2x4* bit2ptr,
                                  utils::bit1x8* bit1ptr, size_t size) {
    return forward<BTLA_ISA::NoSIMD>(srcptr, bit4ptr, bit2ptr, bit1ptr, size);
  }
};

class CompressBit6 {
 public:
  TLACALL BTLA_CODE forward(const int8_t* srcptr, bestla::utils::bit4x2* bit4ptr, utils::bit2x4* bit2ptr, size_t size) {
    return ref::compress_6bit(srcptr, bit4ptr, bit2ptr, size);
  }

  AUTOCALL BTLA_CODE forward_auto(const int8_t* srcptr, bestla::utils::bit4x2* bit4ptr, utils::bit2x4* bit2ptr,
                                  size_t size) {
    return forward<BTLA_ISA::NoSIMD>(srcptr, bit4ptr, bit2ptr, size);
  }
};

class CompressBit5 {
 public:
  TLACALL BTLA_CODE forward(const int8_t* srcptr, bestla::utils::bit4x2* bit4ptr, utils::bit1x8* bit1ptr, size_t size) {
    return ref::compress_5bit(srcptr, bit4ptr, bit1ptr, size);
  }

  AUTOCALL BTLA_CODE forward_auto(const int8_t* srcptr, bestla::utils::bit4x2* bit4ptr, utils::bit1x8* bit1ptr,
                                  size_t size) {
    return forward<BTLA_ISA::NoSIMD>(srcptr, bit4ptr, bit1ptr, size);
  }
};

class CompressBit3 {
 public:
  TLACALL BTLA_CODE forward(const int8_t* srcptr, bestla::utils::bit2x4* bit2ptr, utils::bit1x8* bit1ptr, size_t size) {
    return ref::compress_3bit(srcptr, bit2ptr, bit1ptr, size);
  }

  AUTOCALL BTLA_CODE forward_auto(const int8_t* srcptr, bestla::utils::bit2x4* bit2ptr, utils::bit1x8* bit1ptr,
                                  size_t size) {
    return forward<BTLA_ISA::NoSIMD>(srcptr, bit2ptr, bit1ptr, size);
  }
};

class CompressBit2 {
 public:
  TLACALL BTLA_CODE forward(const int8_t* srcptr, bestla::utils::bit2x4* bit2ptr, size_t size) {
    return ref::compress_2bit(srcptr, bit2ptr, size);
  }

  AUTOCALL BTLA_CODE forward_auto(const int8_t* srcptr, bestla::utils::bit2x4* bit2ptr, size_t size) {
    return forward<BTLA_ISA::NoSIMD>(srcptr, bit2ptr, size);
  }
};

class CompressBit1 {
 public:
  TLACALL BTLA_CODE forward(const int8_t* srcptr, bestla::utils::bit1x8* bit1ptr, size_t size) {
    return ref::compress_1bit(srcptr, bit1ptr, size);
  }

  AUTOCALL BTLA_CODE forward_auto(const int8_t* srcptr, bestla::utils::bit1x8* bit1ptr, size_t size) {
    return forward<BTLA_ISA::NoSIMD>(srcptr, bit1ptr, size);
  }
};

template <typename _T>
class Transpose2D {
 public:
  TLACALL BTLA_CODE forward(const _T* srcptr, _T* dstptr, int row, int col, int ld_src, int ld_dst) {
    return ref::transpose2d(srcptr, dstptr, row, col, ld_src, ld_dst);
  }

  AUTOCALL BTLA_CODE forward_auto(const _T* srcptr, _T* dstptr, int row, int col, int ld_src, int ld_dst) {
    return forward<BTLA_ISA::NoSIMD>(srcptr, dstptr, row, col, ld_src, ld_dst);
  }
};

class QuantizeSignIntRowBlock {
 public:
  TLACALL BTLA_CODE forward(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst,
                            float* scales, int8_t* zero_points, int blocksize, BTLA_DTYPE qtype) {
    // TODO(Yu) simd version for quick quant
    // #if CompileAVX512F()
    //     if constexpr (utils::isa_base<ISA_T>::avx512f) {
    //       return avx512f::quantize_f32_sign_int_rowblock<QDT_T>(srcptr, dstptr, row, col, ld_src, ld_dst, scales,
    //                                                             zero_points, blocksize);
    //     }
    // #endif
    return ref::quantize_f32_sign_int_rowblock(srcptr, dstptr, row, col, ld_src, ld_dst, scales, zero_points, blocksize,
                                               qtype);
  }

  AUTOCALL BTLA_CODE forward_auto(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst,
                                  float* scales, int8_t* zero_points, int blocksize, BTLA_DTYPE qtype) {
    return forward<BTLA_ISA::NoSIMD>(srcptr, dstptr, row, col, ld_src, ld_dst, scales, zero_points, blocksize, qtype);
  }
};

template <BTLA_DTYPE F8_T>
class QuantizeF8RowBlock {
 public:
  TLACALL BTLA_CODE forward(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst,
                            float* scales, int blocksize, BTLA_DTYPE scale_dtype) {
    return ref::quantize_f32_f8_rowblock_mxscale<F8_T>(srcptr, dstptr, row, col, ld_src, ld_dst, scales, blocksize,
                                                       scale_dtype);
  }

  AUTOCALL BTLA_CODE forward_auto(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst,
                                  float* scales, int blocksize, BTLA_DTYPE scale_dtype) {
    return forward<BTLA_ISA::NoSIMD>(srcptr, dstptr, row, col, ld_src, ld_dst, scales, blocksize, scale_dtype);
  }
};

template <BTLA_DTYPE F4_T>
class QuantizeF4RowBlock {
 public:
  TLACALL BTLA_CODE forward(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst,
                            float* scales, int blocksize) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::quantize_f32_f4_rowblock<F4_T>(srcptr, dstptr, row, col, ld_src, ld_dst, scales, blocksize);
    }
#endif
    return ref::quantize_f32_f4_rowblock<F4_T>(srcptr, dstptr, row, col, ld_src, ld_dst, scales, blocksize);
  }

  AUTOCALL BTLA_CODE forward_auto(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst,
                                  float* scales, int blocksize) {
    GetCPUDevice();
    if (_cd->AVX512F()) {
      return forward<BTLA_ISA::AVX512F>(srcptr, dstptr, row, col, ld_src, ld_dst, scales, blocksize);
    }
    return forward<BTLA_ISA::NoSIMD>(srcptr, dstptr, row, col, ld_src, ld_dst, scales, blocksize);
  }
};

template <typename SRC_T>
class QuantizeU8ColBlock {
 public:
  TLACALL BTLA_CODE forward(int row, int col, const SRC_T* srcptr, int ld_src, uint8_t* dstptr, int ld_dst,
                            float* scales, int ld_scale, uint8_t* zps, int blocksize, float* blkreduce) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::quantize_fp_u8_colblock<SRC_T>(row, col, srcptr, ld_src, dstptr, ld_dst, scales, ld_scale, zps,
                                                     blocksize, blkreduce);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return avx2::quantize_fp_u8_colblock<SRC_T>(row, col, srcptr, ld_src, dstptr, ld_dst, scales, ld_scale, zps,
                                                  blocksize, blkreduce);
    }
#endif
    return ref::quantize_fp_u8_colblock(row, col, srcptr, ld_src, dstptr, ld_dst, scales, ld_scale, zps, blocksize,
                                        blkreduce);
  }

  AUTOCALL BTLA_CODE forward_auto(int row, int col, const SRC_T* srcptr, int ld_src, uint8_t* dstptr, int ld_dst,
                                  float* scales, int ld_scale, uint8_t* zps, int blocksize, float* blkreduce) {
    GetCPUDevice();
    if (_cd->AVX512F()) {
      return forward<BTLA_ISA::AVX512F>(row, col, srcptr, ld_src, dstptr, ld_dst, scales, ld_scale, zps, blocksize,
                                        blkreduce);
    }
    if (_cd->AVX2()) {
      return forward<BTLA_ISA::AVX2>(row, col, srcptr, ld_src, dstptr, ld_dst, scales, ld_scale, zps, blocksize,
                                     blkreduce);
    }
    return forward<BTLA_ISA::NoSIMD>(row, col, srcptr, ld_src, dstptr, ld_dst, scales, ld_scale, zps, blocksize,
                                     blkreduce);
  }
};

template <typename SRC_T>
class QuantizeS8ColBlock {
 public:
  TLACALL BTLA_CODE forward(int row, int col, const SRC_T* srcptr, int ld_src, int8_t* dstptr, int ld_dst,
                            float* scales, int ld_scale, int blocksize, float* reduce) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::quantize_fp_s8_colblock<SRC_T>(row, col, srcptr, ld_src, dstptr, ld_dst, scales, ld_scale,
                                                     blocksize, reduce);
    }
#endif
    return ref::quantize_fp_s8_colblock(row, col, srcptr, ld_src, dstptr, ld_dst, scales, ld_scale, blocksize, reduce);
  }

  AUTOCALL BTLA_CODE forward_auto(int row, int col, const SRC_T* srcptr, int ld_src, int8_t* dstptr, int ld_dst,
                                  float* scales, int ld_scale, int blocksize, float* reduce) {
    GetCPUDevice();
    if (_cd->AVX512F()) {
      return forward<BTLA_ISA::AVX512F>(row, col, srcptr, ld_src, dstptr, ld_dst, scales, ld_scale, blocksize, reduce);
    }
    if (_cd->AVX2()) {
      return forward<BTLA_ISA::AVX2>(row, col, srcptr, ld_src, dstptr, ld_dst, scales, ld_scale, blocksize, reduce);
    }
    return forward<BTLA_ISA::NoSIMD>(row, col, srcptr, ld_src, dstptr, ld_dst, scales, ld_scale, blocksize, reduce);
  }
};

class Broadcast {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(int num, const uint8_t& srcval, uint8_t* dstptr) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::broadcast_u8(num, srcval, dstptr);
    }
#endif
    return ref::broadcast_u8(num, srcval, dstptr);
  }
};

class AccumulateDequantizeS32F32 {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(const int32_t* srcptr, float* dstptr, float alpha, float beta, int row, int col,
                                  int ld_src, int ld_dst, float* ascales, int ldas, float* wscales) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::accumulate_dequantize_s32_f32(srcptr, dstptr, alpha, beta, row, col, ld_src, ld_dst, ascales,
                                                    ldas, wscales);
    }
#endif
    return ref::accumulate_dequantize_s32_f32(srcptr, dstptr, alpha, beta, row, col, ld_src, ld_dst, ascales, ldas,
                                              wscales);
  }
};

template <int PackRow, int NTILE>
class DecompressKBlockS4S8 {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(utils::int4x2* srcptr, int8_t* zpptr, int8_t* dstptr, int blocksize, int ldzp,
                                  int n_offset, int k_offset, int row, int col, void* tmp, size_t tmpsize) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f && NTILE % 16 == 0) {
      return avx512f::decompress_kblock_s4_s8<PackRow, NTILE>(srcptr, zpptr, dstptr, blocksize, ldzp, n_offset,
                                                              k_offset, row, col, (int8_t*)tmp, tmpsize);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2 && NTILE % 8 == 0) {
      return avx2::decompress_kblock_s4_s8<PackRow, NTILE>(srcptr, zpptr, dstptr, blocksize, ldzp, n_offset, k_offset,
                                                           row, col, (int8_t*)tmp, tmpsize);
    }
#endif
    return ref::decompress_kblock_s4_s8<PackRow, NTILE>(srcptr, zpptr, dstptr, blocksize, ldzp, n_offset, k_offset, row,
                                                        col, (int8_t*)tmp, tmpsize);
  }
};

template <int PackRow, int NTILE>
class DecompressKBlockS7S8 {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(utils::bit4x2* b4ptr, utils::bit2x4* b2ptr, utils::bit1x8* b1ptr, int8_t* zpptr,
                                  int8_t* dstptr, int blocksize, int ldzp, int n_offset, int k_offset, int row, int col,
                                  void* tmp, size_t tmpsize) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f && NTILE % 16 == 0) {
      return avx512f::decompress_kblock_s7_s8<PackRow, NTILE>(b4ptr, b2ptr, b1ptr, zpptr, dstptr, blocksize, ldzp,
                                                              n_offset, k_offset, row, col, (int8_t*)tmp, tmpsize);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2 && NTILE % 8 == 0) {
      return avx2::decompress_kblock_s7_s8<PackRow, NTILE>(b4ptr, b2ptr, b1ptr, zpptr, dstptr, blocksize, ldzp,
                                                           n_offset, k_offset, row, col, (int8_t*)tmp, tmpsize);
    }
#endif
    return ref::decompress_kblock_s7_s8<PackRow, NTILE>(b4ptr, b2ptr, b1ptr, zpptr, dstptr, blocksize, ldzp, n_offset,
                                                        k_offset, row, col, (int8_t*)tmp, tmpsize);
  }
};

template <int PackRow, int NTILE>
class DecompressKBlockS6S8 {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(utils::bit4x2* b4ptr, utils::bit2x4* b2ptr, int8_t* zpptr, int8_t* dstptr,
                                  int blocksize, int ldzp, int n_offset, int k_offset, int row, int col, void* tmp,
                                  size_t tmpsize) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f && NTILE % 16 == 0) {
      return avx512f::decompress_kblock_s6_s8<PackRow, NTILE>(b4ptr, b2ptr, zpptr, dstptr, blocksize, ldzp, n_offset,
                                                              k_offset, row, col, (int8_t*)tmp, tmpsize);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return avx2::decompress_kblock_s6_s8<PackRow, NTILE>(b4ptr, b2ptr, zpptr, dstptr, blocksize, ldzp, n_offset,
                                                           k_offset, row, col, (int8_t*)tmp, tmpsize);
    }
#endif
    return ref::decompress_kblock_s6_s8<PackRow, NTILE>(b4ptr, b2ptr, zpptr, dstptr, blocksize, ldzp, n_offset,
                                                        k_offset, row, col, (int8_t*)tmp, tmpsize);
  }
};

template <int PackRow, int NTILE>
class DecompressKBlockS5S8 {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(utils::bit4x2* b4ptr, utils::bit1x8* b1ptr, int8_t* zpptr, int8_t* dstptr,
                                  int blocksize, int ldzp, int n_offset, int k_offset, int row, int col, void* tmp,
                                  size_t tmpsize) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f && NTILE % 16 == 0) {
      return avx512f::decompress_kblock_s5_s8<PackRow, NTILE>(b4ptr, b1ptr, zpptr, dstptr, blocksize, ldzp, n_offset,
                                                              k_offset, row, col, (int8_t*)tmp, tmpsize);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return avx2::decompress_kblock_s5_s8<PackRow, NTILE>(b4ptr, b1ptr, zpptr, dstptr, blocksize, ldzp, n_offset,
                                                           k_offset, row, col, (int8_t*)tmp, tmpsize);
    }
#endif
    return ref::decompress_kblock_s5_s8<PackRow, NTILE>(b4ptr, b1ptr, zpptr, dstptr, blocksize, ldzp, n_offset,
                                                        k_offset, row, col, (int8_t*)tmp, tmpsize);
  }
};

template <int PackRow, int NTILE>
class DecompressKBlockS3S8 {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(utils::bit2x4* b2ptr, utils::bit1x8* b1ptr, int8_t* zpptr, int8_t* dstptr,
                                  int blocksize, int ldzp, int n_offset, int k_offset, int row, int col, void* tmp,
                                  size_t tmpsize) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f && NTILE % 16 == 0) {
      return avx512f::decompress_kblock_s3_s8<PackRow, NTILE>(b2ptr, b1ptr, zpptr, dstptr, blocksize, ldzp, n_offset,
                                                              k_offset, row, col, (int8_t*)tmp, tmpsize);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return avx2::decompress_kblock_s3_s8<PackRow, NTILE>(b2ptr, b1ptr, zpptr, dstptr, blocksize, ldzp, n_offset,
                                                           k_offset, row, col, (int8_t*)tmp, tmpsize);
    }
#endif
    return ref::decompress_kblock_s3_s8<PackRow, NTILE>(b2ptr, b1ptr, zpptr, dstptr, blocksize, ldzp, n_offset,
                                                        k_offset, row, col, (int8_t*)tmp, tmpsize);
  }
};

template <int PackRow, int NTILE>
class DecompressKBlockS2S8 {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(utils::bit2x4* b2ptr, int8_t* zpptr, int8_t* dstptr, int blocksize, int ldzp,
                                  int n_offset, int k_offset, int row, int col, void* tmp, size_t tmpsize) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f && NTILE % 16 == 0) {
      return avx512f::decompress_kblock_s2_s8<PackRow, NTILE>(b2ptr, zpptr, dstptr, blocksize, ldzp, n_offset, k_offset,
                                                              row, col, (int8_t*)tmp, tmpsize);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return avx2::decompress_kblock_s2_s8<PackRow, NTILE>(b2ptr, zpptr, dstptr, blocksize, ldzp, n_offset, k_offset,
                                                           row, col, (int8_t*)tmp, tmpsize);
    }
#endif
    return ref::decompress_kblock_s2_s8<PackRow, NTILE>(b2ptr, zpptr, dstptr, blocksize, ldzp, n_offset, k_offset, row,
                                                        col, (int8_t*)tmp, tmpsize);
  }
};

template <int PackRow, int NTILE>
class DecompressKBlockS1S8 {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(utils::bit1x8* b1ptr, int8_t* zpptr, int8_t* dstptr, int blocksize, int ldzp,
                                  int n_offset, int k_offset, int row, int col, void* tmp, size_t tmpsize) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f && NTILE % 16 == 0) {
      return avx512f::decompress_kblock_s1_s8<PackRow, NTILE>(b1ptr, zpptr, dstptr, blocksize, ldzp, n_offset, k_offset,
                                                              row, col, (int8_t*)tmp, tmpsize);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return avx2::decompress_kblock_s1_s8<PackRow, NTILE>(b1ptr, zpptr, dstptr, blocksize, ldzp, n_offset, k_offset,
                                                           row, col, (int8_t*)tmp, tmpsize);
    }
#endif
    return ref::decompress_kblock_s1_s8<PackRow, NTILE>(b1ptr, zpptr, dstptr, blocksize, ldzp, n_offset, k_offset, row,
                                                        col, (int8_t*)tmp, tmpsize);
  }
};

template <int PackRow, int NTILE, typename DstT>
class DecompressKBlockS8Fp {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(int8_t* srcptr, DstT* dstptr, int row, int col, void* scales, BTLA_DTYPE sdtype,
                                  int8_t* zero_points, int k_offset, int n_offset, int kblock, int NPad, void* tmp,
                                  size_t tmpsize) {
    BTLA_CODE ret = BTLA_CODE::NotSupport;
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f && NTILE % 16 == 0) {
      ret = avx512f::decompress_kblock_s8_fp<PackRow, NTILE, DstT>(srcptr, dstptr, row, col, scales, sdtype,
                                                                   zero_points, k_offset, n_offset, kblock, NPad,
                                                                   reinterpret_cast<int8_t*>(tmp), tmpsize);
      if (ret == BTLA_CODE::Success) return ret;
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      ret = avx2::decompress_kblock_s8_fp<PackRow, NTILE, DstT>(srcptr, dstptr, row, col, scales, sdtype, zero_points,
                                                                k_offset, n_offset, kblock, NPad,
                                                                reinterpret_cast<int8_t*>(tmp), tmpsize);
      if (ret == BTLA_CODE::Success) return ret;
    }
#endif
    ret = ref::decompress_kblock_s8_fp<PackRow, NTILE, DstT>(srcptr, dstptr, row, col, scales, sdtype, zero_points,
                                                             k_offset, n_offset, kblock, NPad,
                                                             reinterpret_cast<int8_t*>(tmp), tmpsize);
    return ret;
  }
};

template <int PackRow, int NTILE, typename DstT>
class DecompressKBlockS7Fp {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(utils::bit4x2* b4ptr, utils::bit2x4* b2ptr, utils::bit1x8* b1ptr, DstT* dstptr,
                                  int row, int col, void* scales, BTLA_DTYPE sdtype, int8_t* zero_points, int k_offset,
                                  int n_offset, int kblock, int NPad, void* tmp, size_t tmpsize) {
    BTLA_CODE ret = BTLA_CODE::NotSupport;
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f && NTILE % 16 == 0) {
      return avx512f::decompress_kblock_s7_fp<PackRow, NTILE, DstT>(b4ptr, b2ptr, b1ptr, dstptr, row, col, scales,
                                                                    sdtype, zero_points, k_offset, n_offset, kblock,
                                                                    NPad, reinterpret_cast<int8_t*>(tmp), tmpsize);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return avx2::decompress_kblock_s7_fp<PackRow, NTILE, DstT>(b4ptr, b2ptr, b1ptr, dstptr, row, col, scales, sdtype,
                                                                 zero_points, k_offset, n_offset, kblock, NPad,
                                                                 reinterpret_cast<int8_t*>(tmp), tmpsize);
    }
#endif
    ret = ref::decompress_kblock_s7_fp<PackRow, NTILE, DstT>(b4ptr, b2ptr, b1ptr, dstptr, row, col, scales, sdtype,
                                                             zero_points, k_offset, n_offset, kblock, NPad,
                                                             reinterpret_cast<int8_t*>(tmp), tmpsize);
    return ret;
  }
};

template <int PackRow, int NTILE, typename DstT>
class DecompressKBlockS6Fp {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(utils::bit4x2* b4ptr, utils::bit2x4* b2ptr, DstT* dstptr, int row, int col,
                                  void* scales, BTLA_DTYPE sdtype, int8_t* zero_points, int k_offset, int n_offset,
                                  int kblock, int NPad, void* tmp, size_t tmpsize) {
    BTLA_CODE ret = BTLA_CODE::NotSupport;
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f && NTILE % 16 == 0) {
      return avx512f::decompress_kblock_s6_fp<PackRow, NTILE, DstT>(b4ptr, b2ptr, dstptr, row, col, scales, sdtype,
                                                                    zero_points, k_offset, n_offset, kblock, NPad,
                                                                    reinterpret_cast<int8_t*>(tmp), tmpsize);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return avx2::decompress_kblock_s6_fp<PackRow, NTILE, DstT>(b4ptr, b2ptr, dstptr, row, col, scales, sdtype,
                                                                 zero_points, k_offset, n_offset, kblock, NPad,
                                                                 reinterpret_cast<int8_t*>(tmp), tmpsize);
    }
#endif
    ret = ref::decompress_kblock_s6_fp<PackRow, NTILE, DstT>(b4ptr, b2ptr, dstptr, row, col, scales, sdtype,
                                                             zero_points, k_offset, n_offset, kblock, NPad,
                                                             reinterpret_cast<int8_t*>(tmp), tmpsize);
    return ret;
  }
};

template <int PackRow, int NTILE, typename DstT>
class DecompressKBlockS5Fp {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(utils::bit4x2* b4ptr, utils::bit1x8* b1ptr, DstT* dstptr, int row, int col,
                                  void* scales, BTLA_DTYPE sdtype, int8_t* zero_points, int k_offset, int n_offset,
                                  int kblock, int NPad, void* tmp, size_t tmpsize) {
    BTLA_CODE ret = BTLA_CODE::NotSupport;
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f && NTILE % 16 == 0) {
      return avx512f::decompress_kblock_s5_fp<PackRow, NTILE, DstT>(b4ptr, b1ptr, dstptr, row, col, scales, sdtype,
                                                                    zero_points, k_offset, n_offset, kblock, NPad,
                                                                    reinterpret_cast<int8_t*>(tmp), tmpsize);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return avx2::decompress_kblock_s5_fp<PackRow, NTILE, DstT>(b4ptr, b1ptr, dstptr, row, col, scales, sdtype,
                                                                 zero_points, k_offset, n_offset, kblock, NPad,
                                                                 reinterpret_cast<int8_t*>(tmp), tmpsize);
    }
#endif
    ret = ref::decompress_kblock_s5_fp<PackRow, NTILE, DstT>(b4ptr, b1ptr, dstptr, row, col, scales, sdtype,
                                                             zero_points, k_offset, n_offset, kblock, NPad,
                                                             reinterpret_cast<int8_t*>(tmp), tmpsize);
    return ret;
  }
};

template <int PackRow, int NTILE, typename DstT>
class DecompressKBlockS4Fp {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(utils::int4x2* srcptr, DstT* dstptr, int row, int col, void* scales,
                                  BTLA_DTYPE sdtype, int8_t* zero_points, int k_offset, int n_offset, int kblock,
                                  int NPad, void* tmp, size_t tmpsize) {
    BTLA_CODE ret = BTLA_CODE::NotSupport;
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f && NTILE % 16 == 0) {
      return avx512f::decompress_kblock_s4_fp<PackRow, NTILE, DstT>(srcptr, dstptr, row, col, scales, sdtype,
                                                                    zero_points, k_offset, n_offset, kblock, NPad,
                                                                    reinterpret_cast<int8_t*>(tmp), tmpsize);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return avx2::decompress_kblock_s4_fp<PackRow, NTILE, DstT>(srcptr, dstptr, row, col, scales, sdtype, zero_points,
                                                                 k_offset, n_offset, kblock, NPad,
                                                                 reinterpret_cast<int8_t*>(tmp), tmpsize);
    }
#endif
    ret = ref::decompress_kblock_s4_fp<PackRow, NTILE, DstT>(srcptr, dstptr, row, col, scales, sdtype, zero_points,
                                                             k_offset, n_offset, kblock, NPad,
                                                             reinterpret_cast<int8_t*>(tmp), tmpsize);
    return ret;
  }
};

template <int PackRow, int NTILE, typename DstT>
class DecompressKBlockS3Fp {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(utils::bit2x4* b2ptr, utils::bit1x8* b1ptr, DstT* dstptr, int row, int col,
                                  void* scales, BTLA_DTYPE sdtype, int8_t* zero_points, int k_offset, int n_offset,
                                  int kblock, int NPad, void* tmp, size_t tmpsize) {
    BTLA_CODE ret = BTLA_CODE::NotSupport;
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f && NTILE % 16 == 0) {
      return avx512f::decompress_kblock_s3_fp<PackRow, NTILE, DstT>(b2ptr, b1ptr, dstptr, row, col, scales, sdtype,
                                                                    zero_points, k_offset, n_offset, kblock, NPad,
                                                                    reinterpret_cast<int8_t*>(tmp), tmpsize);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return avx2::decompress_kblock_s3_fp<PackRow, NTILE, DstT>(b2ptr, b1ptr, dstptr, row, col, scales, sdtype,
                                                                 zero_points, k_offset, n_offset, kblock, NPad,
                                                                 reinterpret_cast<int8_t*>(tmp), tmpsize);
    }
#endif
    ret = ref::decompress_kblock_s3_fp<PackRow, NTILE, DstT>(b2ptr, b1ptr, dstptr, row, col, scales, sdtype,
                                                             zero_points, k_offset, n_offset, kblock, NPad,
                                                             reinterpret_cast<int8_t*>(tmp), tmpsize);
    return ret;
  }
};

template <int PackRow, int NTILE, typename DstT>
class DecompressKBlockS2Fp {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(utils::bit2x4* b2ptr, DstT* dstptr, int row, int col, void* scales, BTLA_DTYPE sdtype,
                                  int8_t* zero_points, int k_offset, int n_offset, int kblock, int NPad, void* tmp,
                                  size_t tmpsize) {
    BTLA_CODE ret = BTLA_CODE::NotSupport;
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f && NTILE % 16 == 0) {
      return avx512f::decompress_kblock_s2_fp<PackRow, NTILE, DstT>(b2ptr, dstptr, row, col, scales, sdtype,
                                                                    zero_points, k_offset, n_offset, kblock, NPad,
                                                                    reinterpret_cast<int8_t*>(tmp), tmpsize);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return avx2::decompress_kblock_s2_fp<PackRow, NTILE, DstT>(b2ptr, dstptr, row, col, scales, sdtype, zero_points,
                                                                 k_offset, n_offset, kblock, NPad,
                                                                 reinterpret_cast<int8_t*>(tmp), tmpsize);
    }
#endif
    ret = ref::decompress_kblock_s2_fp<PackRow, NTILE, DstT>(b2ptr, dstptr, row, col, scales, sdtype, zero_points,
                                                             k_offset, n_offset, kblock, NPad,
                                                             reinterpret_cast<int8_t*>(tmp), tmpsize);
    return ret;
  }
};

template <int PackRow, int NTILE, typename DstT>
class DecompressKBlockS1Fp {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(utils::bit1x8* b1ptr, DstT* dstptr, int row, int col, void* scales, BTLA_DTYPE sdtype,
                                  int8_t* zero_points, int k_offset, int n_offset, int kblock, int NPad, void* tmp,
                                  size_t tmpsize) {
    BTLA_CODE ret = BTLA_CODE::NotSupport;
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f && NTILE % 16 == 0) {
      return avx512f::decompress_kblock_s1_fp<PackRow, NTILE, DstT>(b1ptr, dstptr, row, col, scales, sdtype,
                                                                    zero_points, k_offset, n_offset, kblock, NPad,
                                                                    reinterpret_cast<int8_t*>(tmp), tmpsize);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return avx2::decompress_kblock_s1_fp<PackRow, NTILE, DstT>(b1ptr, dstptr, row, col, scales, sdtype, zero_points,
                                                                 k_offset, n_offset, kblock, NPad,
                                                                 reinterpret_cast<int8_t*>(tmp), tmpsize);
    }
#endif
    ret = ref::decompress_kblock_s1_fp<PackRow, NTILE, DstT>(b1ptr, dstptr, row, col, scales, sdtype, zero_points,
                                                             k_offset, n_offset, kblock, NPad,
                                                             reinterpret_cast<int8_t*>(tmp), tmpsize);
    return ret;
  }
};

template <typename _DST_T, int _PACK_ROW>
class DecompressKBlockF4Fp {
 public:
  template <BTLA_ISA ISA_T, typename SCA_T, BTLA_DTYPE F4_T>
  static inline BTLA_CODE forward(utils::f4x2* srcptr, _DST_T* dstptr, int row, int col, int ld_src, int ld_dst,
                                  SCA_T* scales, int k_offset, int kblock, int NPad, void* tmp, size_t tmpsize) {
    BTLA_CODE ret = BTLA_CODE::NotSupport;
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      ret = avx512f::decompress_kblock_f4_fp<F4_T, _DST_T, _PACK_ROW, SCA_T>(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                                             scales, k_offset, kblock, NPad,
                                                                             reinterpret_cast<int8_t*>(tmp), tmpsize);
      if (ret == BTLA_CODE::Success) return ret;
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2 && std::is_same_v<SCA_T, float>) {
      ret = avx2::decompress_kblock_f4_fp<F4_T, _DST_T, _PACK_ROW, SCA_T>(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                                          scales, k_offset, kblock, NPad,
                                                                          reinterpret_cast<int8_t*>(tmp), tmpsize);
      if (ret == BTLA_CODE::Success) return ret;
    }
#endif
    return ref::decompress_kblock_f4_fp<F4_T, _DST_T, _PACK_ROW, SCA_T>(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                                        scales, k_offset, kblock, NPad,
                                                                        reinterpret_cast<int8_t*>(tmp), tmpsize);
  }
};

template <typename _DST_T, int _PACK_ROW>
class DecompressDqKBlockF4Fp {
 public:
  template <BTLA_ISA ISA_T, BTLA_DTYPE F4_T, typename SCA_T>
  static inline BTLA_CODE forward(utils::f4x2* srcptr, _DST_T* dstptr, int row, int col, int ld_src, int ld_dst,
                                  SCA_T* scales, float* dq_scale, int k_offset, int n_offset, int kblock, int dq_blk,
                                  int dq_offset_idx, int NPad, int N, void* tmp, size_t tmpsize) {
    return ref::decompress_dq_kblock_f4_fp<F4_T, _PACK_ROW>(srcptr, dstptr, row, col, ld_src, ld_dst, scales, dq_scale,
                                                            k_offset, n_offset, kblock, dq_blk, dq_offset_idx, NPad, N,
                                                            tmp, tmpsize);
  }
};

template <typename _DST_T>
class DecompressKBlockF4FpNoscale {
 public:
  template <BTLA_ISA ISA_T, BTLA_DTYPE F4_T>
  static inline BTLA_CODE forward(utils::f4x2* srcptr, _DST_T* dstptr, int row, int col, int ld_src, int ld_dst,
                                  void* tmp, size_t tmpsize) {
    BTLA_CODE ret = BTLA_CODE::NotSupport;
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::decompress_kblock_f4_fp_noscale<F4_T, _DST_T>(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                                    reinterpret_cast<int8_t*>(tmp), tmpsize);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return avx2::decompress_kblock_f4_fp_noscale<F4_T, _DST_T>(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                                 reinterpret_cast<int8_t*>(tmp), tmpsize);
    }
#endif
    return ref::decompress_kblock_f4_fp_noscale<F4_T, _DST_T>(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                              reinterpret_cast<int8_t*>(tmp), tmpsize);
  }
};

template <int PACK_ROW>
class DecompressKBlockF8FP {
 public:
  template <BTLA_ISA ISA_T, typename SCA_T, typename DST_T>
  static inline BTLA_CODE forward(utils::f8* srcptr, DST_T* dstptr, int row, int col, int ld_src, int ld_dst,
                                  SCA_T* scales, int k_offset, int kblock, int NPad, BTLA_DTYPE src_f8_type) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::decompress_kblock_f8_fp<true, DST_T, PACK_ROW, SCA_T>(
          srcptr, dstptr, row, col, ld_src, ld_dst, scales, k_offset, kblock, NPad, src_f8_type);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return avx2::decompress_kblock_f8_fp<true, DST_T, PACK_ROW, SCA_T>(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                                         scales, k_offset, kblock, NPad, src_f8_type);
    }
#endif
    return ref::decompress_kblock_f8_fp<DST_T, PACK_ROW, SCA_T>(srcptr, dstptr, row, col, ld_src, ld_dst, scales,
                                                                k_offset, kblock, NPad, src_f8_type);
  }

  template <typename SCA_T, typename DST_T>
  static inline BTLA_CODE forward_auto(utils::f8* srcptr, DST_T* dstptr, int row, int col, int ld_src, int ld_dst,
                                       SCA_T* scales, int k_offset, int kblock, int NPad, BTLA_DTYPE src_f8_type) {
    GetCPUDevice();
    if (_cd->AVX512F()) {
      return forward<BTLA_ISA::AVX512F>(srcptr, dstptr, row, col, ld_src, ld_dst, scales, k_offset, kblock, NPad,
                                        src_f8_type);
    }
    if (_cd->AVX2()) {
      return forward<BTLA_ISA::AVX2>(srcptr, dstptr, row, col, ld_src, ld_dst, scales, k_offset, kblock, NPad,
                                     src_f8_type);
    }
    return forward<BTLA_ISA::NoSIMD>(srcptr, dstptr, row, col, ld_src, ld_dst, scales, k_offset, kblock, NPad,
                                     src_f8_type);
  }
};

template <typename _DST_T>
class DecompressKBlockF8FpNoScale {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(utils::f8* srcptr, _DST_T* dstptr, int row, int col, int ld_src, int ld_dst,
                                  void* tmp, size_t tmpsize, BTLA_DTYPE src_f8_t) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::decompress_kblock_f8_fp<false, _DST_T, 1>(
          srcptr, dstptr, row, col, ld_src, ld_dst, reinterpret_cast<utils::f8*>(tmp), -1, -1, -1, src_f8_t);
    }
#endif
#if CompileAVX2()
    if (utils::isa_base<ISA_T>::avx2) {
      return avx2::decompress_kblock_f8_fp<false, _DST_T, 1>(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                             reinterpret_cast<utils::f8*>(tmp), -1, -1, -1, src_f8_t);
    }
#endif
    return ref::decompress_kblock_f8_fp_noscale<_DST_T>(srcptr, dstptr, row, col, ld_src, ld_dst, src_f8_t);
  }
};

class AlphaBetaF32F32 {
 public:
  template <BTLA_ISA ISA_T>
  static BTLA_CODE forward(const float alpha, const float* srcptr, const int srcstep, const float beta,
                           const float* src1ptr, const int src1step, float* dstptr, const int dststep, const int M,
                           const int N) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::alphabeta_f32_f32(alpha, srcptr, srcstep, beta, src1ptr, src1step, dstptr, dststep, M, N);
    }
#endif
#if CompileAVX2()
    if (utils::isa_base<ISA_T>::avx2) {
      return avx2::alphabeta_f32_f32(alpha, srcptr, srcstep, beta, src1ptr, src1step, dstptr, dststep, M, N);
    }
#endif
    return ref::alphabeta_f32_f32(alpha, srcptr, srcstep, beta, src1ptr, src1step, dstptr, dststep, M, N);
  }
};

class CompFp32BlockScale {
 public:
  template <BTLA_ISA ISA_T, typename SCA_T>
  static BTLA_CODE forward(const SCA_T* alpha, const float* srcptr, const int srcstep, float* dstptr, const int dststep,
                           const int M, const int N) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::accum_alphaN_f32_f32(alpha, srcptr, srcstep, dstptr, dststep, M, N);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return avx2::accum_alphaN_f32_f32(alpha, srcptr, srcstep, dstptr, dststep, M, N);
    }
#endif
    return ref::accum_alphaN_f32_f32(alpha, srcptr, srcstep, dstptr, dststep, M, N);
  }
};

class AccumulateFp32 {
 public:
  template <BTLA_ISA ISA_T>
  static BTLA_CODE forward(const float* srcptr, const int srcstep, float* dstptr, const int dststep, const int M,
                           const int N) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::accum_f32_f32(srcptr, srcstep, dstptr, dststep, M, N);
    }
#endif
    return ref::accum_f32_f32(srcptr, srcstep, dstptr, dststep, M, N);
  }
};

class QuanOutS32U32 {
 public:
  template <BTLA_ISA ISA_T>
  static BTLA_CODE forward(const float alpha, const int32_t* srcptr, const int srcstep, uint8_t* dstptr,
                           const int dststep, const int M, const int N, float scaleSrc, float scaleDst, int zpDst) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::quanout_s32_u32(alpha, srcptr, srcstep, dstptr, dststep, M, N, scaleSrc, scaleDst, zpDst);
    }
#endif
    return ref::quanout_s32_u32(alpha, srcptr, srcstep, dstptr, dststep, M, N, scaleSrc, scaleDst, zpDst);
  }
};

// scaleA ldsa==0 per tensor, ldsa!=0 per M
// scaleB per channel(N)
class DequanS32Fp32 {
 public:
  template <BTLA_ISA ISA_T, typename SCAB_T>
  static BTLA_CODE forward(const int32_t* srcptr, const int srcstep, float* dstptr, const int dststep, const int M,
                           const int N, const float* scaleA, const int ldsa, const SCAB_T* scaleB) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::dequant_s32_fp32(srcptr, srcstep, dstptr, dststep, M, N, scaleA, ldsa, scaleB);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return avx2::dequant_s32_fp32(srcptr, srcstep, dstptr, dststep, M, N, scaleA, ldsa, scaleB);
    }
#endif
    return ref::dequant_s32_fp32(srcptr, srcstep, dstptr, dststep, M, N, scaleA, ldsa, scaleB);
  }
};

class MinMaxKBlock {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(const float* srcptr, int row, int col, int ld_src, float* minmaxptr, int ld_minmax,
                                  int fsize_minmax, int blocksize) {
    return ref::minmax_f32_kblock(srcptr, row, col, ld_src, minmaxptr, ld_minmax, fsize_minmax, blocksize);
  }
};

template <typename _RT>
class QuantS8RowReduceSum {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(const int8_t* srcptr, int ldsrc, const float* scales, const int8_t* zero_points,
                                  int row, int col, _RT* reduce) {
    return ref::quant_s8_row_reduce_sum(srcptr, ldsrc, scales, zero_points, row, col, reduce);
  }
};

template <typename _SRCT, typename _DSTT>
class RowReduceSum {
 public:
  TLACALL BTLA_CODE forward(const _SRCT* srcptr, int ldsrc, int row, int col, _DSTT* reduce) {
    return ref::row_reduce_sum(srcptr, ldsrc, row, col, reduce);
  }
  AUTOCALL BTLA_CODE forward_auto(const _SRCT* srcptr, int ldsrc, int row, int col, _DSTT* reduce) {
    return forward<BTLA_ISA::NoSIMD>(srcptr, ldsrc, row, col, reduce);
  }
};

template <typename SRC_T>
class ColBlockReduceSum {
 public:
  TLACALL BTLA_CODE forward(const SRC_T* srcptr, int ldsrc, int row, int col, int blocksize, float* reduce, int ldr) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f && std::is_same_v<SRC_T, float>) {
      return avx512f::col_block_reduce_sum<SRC_T>(srcptr, ldsrc, row, col, blocksize, reduce, ldr);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2 && std::is_same_v<SRC_T, float>) {
      return avx2::col_block_reduce_sum<SRC_T>(srcptr, ldsrc, row, col, blocksize, reduce, ldr);
    }
#endif
    return ref::col_block_reduce_sum<SRC_T>(srcptr, ldsrc, row, col, blocksize, reduce, ldr);
  }

  AUTOCALL BTLA_CODE forward_auto(const SRC_T* srcptr, int ldsrc, int row, int col, int blocksize, float* reduce,
                                  int ldr) {
    GetCPUDevice();
    if (_cd->AVX512F()) {
      return forward<BTLA_ISA::AVX512F>(srcptr, ldsrc, row, col, blocksize, reduce, ldr);
    }
    if (_cd->AVX2()) {
      return forward<BTLA_ISA::AVX2>(srcptr, ldsrc, row, col, blocksize, reduce, ldr);
    }
    return forward<BTLA_ISA::NoSIMD>(srcptr, ldsrc, row, col, blocksize, reduce, ldr);
  }
};
class RemoveZeroPointBias {
 public:
  template <BTLA_ISA ISA_T, bool ZPA, bool ZPB>
  static inline BTLA_CODE forward_inner(int32_t* accptr, int ldacc, int row, int col, uint8_t* zpa, int8_t* zpb,
                                        int lda, int k, const int32_t* reducea, const int32_t* reduceb) {
    if constexpr (ZPA) {
      assert(zpa);
    }
    if constexpr (ZPB) {
      assert(zpb);
    }
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::remove_zeropoint_bias<ZPA, ZPB>(accptr, ldacc, row, col, zpa, zpb, lda, k, reducea, reduceb);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return avx2::remove_zeropoint_bias<ZPA, ZPB>(accptr, ldacc, row, col, zpa, zpb, lda, k, reducea, reduceb);
    }
#endif
    return ref::remove_zeropoint_bias<ZPA, ZPB>(accptr, ldacc, row, col, zpa, zpb, lda, k, reducea, reduceb);
  }

  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(int32_t* accptr, int ldacc, int row, int col, uint8_t* zpa, int8_t* zpb, int lda,
                                  int k, const int32_t* reducea, const int32_t* reduceb) {
    if (zpa) {
      if (zpb) {
        return forward_inner<ISA_T, true, true>(accptr, ldacc, row, col, zpa, zpb, lda, k, reducea, reduceb);
      } else {
        return forward_inner<ISA_T, true, false>(accptr, ldacc, row, col, zpa, zpb, lda, k, reducea, reduceb);
      }
    } else {
      if (zpb) {
        return forward_inner<ISA_T, false, true>(accptr, ldacc, row, col, zpa, zpb, lda, k, reducea, reduceb);
      } else {
        assert(0);
      }
    }
    assert(0);
    return BTLA_CODE::NotSupport;
  }
};

class LayerNormalization {
 public:
  template <BTLA_ISA ISA_T, typename T>
  static inline BTLA_CODE forward(const T* srcptr, const T* scaleptr, const T* biasptr, T epsilon, int norm_size,
                                  T* dstptr, T* mean, T* mean_square, bool simplified) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f && std::is_same_v<T, float>) {
      return avx512f::layernorm(srcptr, scaleptr, biasptr, epsilon, norm_size, dstptr, mean, mean_square, simplified);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2 && std::is_same_v<T, float>) {
      return avx2::layernorm(srcptr, scaleptr, biasptr, epsilon, norm_size, dstptr, mean, mean_square, simplified);
    }
#endif
    return ref::layernorm(srcptr, scaleptr, biasptr, epsilon, norm_size, dstptr, mean, mean_square, simplified);
  }
  template <typename T>
  static inline BTLA_CODE forward_auto(const T* srcptr, const T* scaleptr, const T* biasptr, T epsilon, int norm_size,
                                       T* dstptr, T* mean, T* mean_square, bool simplified) {
    GetCPUDevice();
    if (_cd->AVX512F()) {
      return forward<BTLA_ISA::AVX512F, T>(srcptr, scaleptr, biasptr, epsilon, norm_size, dstptr, mean, mean_square,
                                           simplified);
    }
    if (_cd->AVX2()) {
      return forward<BTLA_ISA::AVX2, T>(srcptr, scaleptr, biasptr, epsilon, norm_size, dstptr, mean, mean_square,
                                        simplified);
    }
    return forward<BTLA_ISA::NoSIMD, T>(srcptr, scaleptr, biasptr, epsilon, norm_size, dstptr, mean, mean_square,
                                        simplified);
  }
};

class GEMVWoqNBits {
 public:
  template <BTLA_ISA ISA_T, typename ScaleT, int NTILE, int MTILE>
  static inline BTLA_CODE forward_u8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, void* tmp, size_t tmpsize) {
    if (B.nbits == 6) {
#if CompileAVX512VNNI()
      if (ISA_T >= BTLA_ISA::AVX512_VNNI && NTILE % 16 == 0) {
        return avx512f::vnni::gemv_6bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp,
                                                                        tmpsize);
      }
#endif
#if CompileAVX512F()
      if (ISA_T >= BTLA_ISA::AVX512BW && NTILE % 16 == 0) {
        return avx512f::gemv_6bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
#if CompileAVXVNNI()
      if (ISA_T >= BTLA_ISA::AVX_VNNI) {
        return avx2::vnni::gemv_6bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
#if CompileAVX2()
      if (ISA_T >= BTLA_ISA::AVX2) {
        return avx2::gemv_6bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
      return ref::gemv_6bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
    }
    if (B.nbits == 5) {
#if CompileAVX512VNNI()
      if (ISA_T >= BTLA_ISA::AVX512_VNNI && NTILE % 16 == 0) {
        return avx512f::vnni::gemv_5bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp,
                                                                        tmpsize);
      }
#endif
#if CompileAVX512F()
      if (ISA_T >= BTLA_ISA::AVX512BW && NTILE % 16 == 0) {
        return avx512f::gemv_5bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
#if CompileAVXVNNI()
      if (ISA_T >= BTLA_ISA::AVX_VNNI) {
        return avx2::vnni::gemv_5bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
#if CompileAVX2()
      if (ISA_T >= BTLA_ISA::AVX2) {
        return avx2::gemv_5bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
      return ref::gemv_5bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
    }
    if (B.nbits == 4) {
#if CompileAVX512VNNI()
      if (ISA_T >= BTLA_ISA::AVX512_VNNI && NTILE % 16 == 0) {
        return avx512f::vnni::gemv_4bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp,
                                                                        tmpsize);
      }
#endif
#if CompileAVX512F()
      if (ISA_T >= BTLA_ISA::AVX512BW && NTILE % 16 == 0) {
        return avx512f::gemv_4bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
#if CompileAVXVNNI()
      if (ISA_T >= BTLA_ISA::AVX_VNNI) {
        return avx2::vnni::gemv_4bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
#if CompileAVX2()
      if (ISA_T >= BTLA_ISA::AVX2) {
        return avx2::gemv_4bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
      return ref::gemv_4bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
    }
    if (B.nbits == 3) {
#if CompileAVX512VNNI()
      if (ISA_T >= BTLA_ISA::AVX512_VNNI && NTILE % 16 == 0) {
        return avx512f::vnni::gemv_3bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp,
                                                                        tmpsize);
      }
#endif
#if CompileAVX512F()
      if (ISA_T >= BTLA_ISA::AVX512BW && NTILE % 16 == 0) {
        return avx512f::gemv_3bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
#if CompileAVXVNNI()
      if (ISA_T >= BTLA_ISA::AVX_VNNI) {
        return avx2::vnni::gemv_3bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
#if CompileAVX2()
      if (ISA_T >= BTLA_ISA::AVX2) {
        return avx2::gemv_3bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
      return ref::gemv_3bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
    }
    if (B.nbits == 2) {
#if CompileAVX512VNNI()
      if (ISA_T >= BTLA_ISA::AVX512_VNNI && NTILE % 16 == 0) {
        return avx512f::vnni::gemv_2bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp,
                                                                        tmpsize);
      }
#endif
#if CompileAVX512F()
      if (ISA_T >= BTLA_ISA::AVX512BW && NTILE % 16 == 0) {
        return avx512f::gemv_2bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
#if CompileAVXVNNI()
      if (ISA_T >= BTLA_ISA::AVX_VNNI) {
        return avx2::vnni::gemv_2bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
#if CompileAVX2()
      if (ISA_T >= BTLA_ISA::AVX2) {
        return avx2::gemv_2bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
      return ref::gemv_2bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
    }
    if (B.nbits == 7) {
#if CompileAVX512VNNI()
      if (ISA_T >= BTLA_ISA::AVX512_VNNI && NTILE % 16 == 0) {
        return avx512f::vnni::gemv_7bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp,
                                                                        tmpsize);
      }
#endif
#if CompileAVX512F()
      if (ISA_T >= BTLA_ISA::AVX512BW && NTILE % 16 == 0) {
        return avx512f::gemv_7bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
#if CompileAVXVNNI()
      if (ISA_T >= BTLA_ISA::AVX_VNNI) {
        return avx2::vnni::gemv_7bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
#if CompileAVX2()
      if (ISA_T >= BTLA_ISA::AVX2) {
        return avx2::gemv_7bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
      return ref::gemv_7bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
    }
    if (B.nbits == 1) {
#if CompileAVX512VNNI()
      if (ISA_T >= BTLA_ISA::AVX512_VNNI && NTILE % 16 == 0) {
        return avx512f::vnni::gemv_1bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp,
                                                                        tmpsize);
      }
#endif
#if CompileAVX512F()
      if (ISA_T >= BTLA_ISA::AVX512BW && NTILE % 16 == 0) {
        return avx512f::gemv_1bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
#if CompileAVXVNNI()
      if (ISA_T >= BTLA_ISA::AVX_VNNI) {
        return avx2::vnni::gemv_1bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
#if CompileAVX2()
      if (ISA_T >= BTLA_ISA::AVX2) {
        return avx2::gemv_1bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
      return ref::gemv_1bit_u8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
    }
    return BTLA_CODE::NotSupport;
  }

  template <BTLA_ISA ISA_T, typename ScaleT, int NTILE, int MTILE>
  static inline BTLA_CODE forward_s8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, void* tmp, size_t tmpsize) {
    if (B.nbits == 5) {
#if CompileAVX512VNNI()
      if (ISA_T >= BTLA_ISA::AVX512_VNNI && NTILE % 16 == 0) {
        return avx512f::vnni::gemv_5bit_s8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp,
                                                                        tmpsize);
      }
#endif
#if CompileAVXVNNI()
      if (ISA_T >= BTLA_ISA::AVX_VNNI) {
        return avx2::vnni::gemv_5bit_s8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
      return ref::gemv_5bit_s8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
    }
    if (B.nbits == 4) {
#if CompileAVX512VNNI()
      if (ISA_T >= BTLA_ISA::AVX512_VNNI && NTILE % 16 == 0) {
        return avx512f::vnni::gemv_4bit_s8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp,
                                                                        tmpsize);
      }
#endif
#if CompileAVXVNNI()
      if (ISA_T >= BTLA_ISA::AVX_VNNI) {
        return avx2::vnni::gemv_4bit_s8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
      return ref::gemv_4bit_s8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
    }
    if (B.nbits == 3) {
#if CompileAVX512VNNI()
      if (ISA_T >= BTLA_ISA::AVX512_VNNI && NTILE % 16 == 0) {
        return avx512f::vnni::gemv_3bit_s8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp,
                                                                        tmpsize);
      }
#endif
#if CompileAVXVNNI()
      if (ISA_T >= BTLA_ISA::AVX_VNNI) {
        return avx2::vnni::gemv_3bit_s8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
      return ref::gemv_3bit_s8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
    }
    if (B.nbits == 2) {
#if CompileAVX512VNNI()
      if (ISA_T >= BTLA_ISA::AVX512_VNNI && NTILE % 16 == 0) {
        return avx512f::vnni::gemv_2bit_s8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp,
                                                                        tmpsize);
      }
#endif
#if CompileAVXVNNI()
      if (ISA_T >= BTLA_ISA::AVX_VNNI) {
        return avx2::vnni::gemv_2bit_s8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
      return ref::gemv_2bit_s8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
    }
    if (B.nbits == 6) {
#if CompileAVX512VNNI()
      if (ISA_T >= BTLA_ISA::AVX512_VNNI && NTILE % 16 == 0) {
        return avx512f::vnni::gemv_6bit_s8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp,
                                                                        tmpsize);
      }
#endif
#if CompileAVXVNNI()
      if (ISA_T >= BTLA_ISA::AVX_VNNI) {
        return avx2::vnni::gemv_6bit_s8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
      return ref::gemv_6bit_s8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
    }
    if (B.nbits == 7) {
#if CompileAVX512VNNI()
      if (ISA_T >= BTLA_ISA::AVX512_VNNI && NTILE % 16 == 0) {
        return avx512f::vnni::gemv_7bit_s8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp,
                                                                        tmpsize);
      }
#endif
#if CompileAVXVNNI()
      if (ISA_T >= BTLA_ISA::AVX_VNNI) {
        return avx2::vnni::gemv_7bit_s8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
      return ref::gemv_7bit_s8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
    }
    if (B.nbits == 1) {
#if CompileAVX512VNNI()
      if (ISA_T >= BTLA_ISA::AVX512_VNNI && NTILE % 16 == 0) {
        return avx512f::vnni::gemv_1bit_s8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp,
                                                                        tmpsize);
      }
#endif
#if CompileAVXVNNI()
      if (ISA_T >= BTLA_ISA::AVX_VNNI) {
        return avx2::vnni::gemv_1bit_s8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
      return ref::gemv_1bit_s8s8_fp32<ScaleT, NTILE, MTILE>(A, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
    }
    return BTLA_CODE::NotSupport;
  }

  template <BTLA_ISA ISA_T, typename ScaleT, int NTILE, int MTILE>
  static inline BTLA_CODE forward_fp32_fp32(const float* A, int lda, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, void* tmp, size_t tmpsize) {
    if (B.nbits == 6) {
#if CompileAVX512F()
      if (ISA_T >= BTLA_ISA::AVX512F && NTILE % 16 == 0) {
        return avx512f::gemv_6bit_fp32_fp32<ScaleT, NTILE, MTILE>(A, lda, B, C, ldc, k, blocksize, (int8_t*)tmp,
                                                                  tmpsize);
      }
#endif
#if CompileAVX2()
      if (ISA_T >= BTLA_ISA::AVX2) {
        return avx2::gemv_6bit_fp32_fp32<ScaleT, NTILE, MTILE>(A, lda, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
      return ref::gemv_6bit_fp32_fp32<ScaleT, NTILE, MTILE>(A, lda, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
    }
    if (B.nbits == 5) {
#if CompileAVX512F()
      if (ISA_T >= BTLA_ISA::AVX512F && NTILE % 16 == 0) {
        return avx512f::gemv_5bit_fp32_fp32<ScaleT, NTILE, MTILE>(A, lda, B, C, ldc, k, blocksize, (int8_t*)tmp,
                                                                  tmpsize);
      }
#endif
#if CompileAVX2()
      if (ISA_T >= BTLA_ISA::AVX2) {
        return avx2::gemv_5bit_fp32_fp32<ScaleT, NTILE, MTILE>(A, lda, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
      return ref::gemv_5bit_fp32_fp32<ScaleT, NTILE, MTILE>(A, lda, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
    }
    if (B.nbits == 4) {
#if CompileAVX512F()
      if (ISA_T >= BTLA_ISA::AVX512F && NTILE % 16 == 0) {
        return avx512f::gemv_4bit_fp32_fp32<ScaleT, NTILE, MTILE>(A, lda, B, C, ldc, k, blocksize, (int8_t*)tmp,
                                                                  tmpsize);
      }
#endif
#if CompileAVX2()
      if (ISA_T >= BTLA_ISA::AVX2) {
        return avx2::gemv_4bit_fp32_fp32<ScaleT, NTILE, MTILE>(A, lda, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
      return ref::gemv_4bit_fp32_fp32<ScaleT, NTILE, MTILE>(A, lda, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
    }
    if (B.nbits == 3) {
#if CompileAVX512F()
      if (ISA_T >= BTLA_ISA::AVX512F && NTILE % 16 == 0) {
        return avx512f::gemv_3bit_fp32_fp32<ScaleT, NTILE, MTILE>(A, lda, B, C, ldc, k, blocksize, (int8_t*)tmp,
                                                                  tmpsize);
      }
#endif
#if CompileAVX2()
      if (ISA_T >= BTLA_ISA::AVX2) {
        return avx2::gemv_3bit_fp32_fp32<ScaleT, NTILE, MTILE>(A, lda, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
      return ref::gemv_3bit_fp32_fp32<ScaleT, NTILE, MTILE>(A, lda, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
    }
    if (B.nbits == 2) {
#if CompileAVX512F()
      if (ISA_T >= BTLA_ISA::AVX512F && NTILE % 16 == 0) {
        return avx512f::gemv_2bit_fp32_fp32<ScaleT, NTILE, MTILE>(A, lda, B, C, ldc, k, blocksize, (int8_t*)tmp,
                                                                  tmpsize);
      }
#endif
#if CompileAVX2()
      if (ISA_T >= BTLA_ISA::AVX2) {
        return avx2::gemv_2bit_fp32_fp32<ScaleT, NTILE, MTILE>(A, lda, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
      return ref::gemv_2bit_fp32_fp32<ScaleT, NTILE, MTILE>(A, lda, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
    }
    if (B.nbits == 7) {
#if CompileAVX512F()
      if (ISA_T >= BTLA_ISA::AVX512F && NTILE % 16 == 0) {
        return avx512f::gemv_7bit_fp32_fp32<ScaleT, NTILE, MTILE>(A, lda, B, C, ldc, k, blocksize, (int8_t*)tmp,
                                                                  tmpsize);
      }
#endif
#if CompileAVX2()
      if (ISA_T >= BTLA_ISA::AVX2) {
        return avx2::gemv_7bit_fp32_fp32<ScaleT, NTILE, MTILE>(A, lda, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
      return ref::gemv_7bit_fp32_fp32<ScaleT, NTILE, MTILE>(A, lda, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
    }
    if (B.nbits == 1) {
#if CompileAVX512F()
      if (ISA_T >= BTLA_ISA::AVX512F && NTILE % 16 == 0) {
        return avx512f::gemv_1bit_fp32_fp32<ScaleT, NTILE, MTILE>(A, lda, B, C, ldc, k, blocksize, (int8_t*)tmp,
                                                                  tmpsize);
      }
#endif
#if CompileAVX2()
      if (ISA_T >= BTLA_ISA::AVX2) {
        return avx2::gemv_1bit_fp32_fp32<ScaleT, NTILE, MTILE>(A, lda, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
      }
#endif
      return ref::gemv_1bit_fp32_fp32<ScaleT, NTILE, MTILE>(A, lda, B, C, ldc, k, blocksize, (int8_t*)tmp, tmpsize);
    }
    return BTLA_CODE::NotSupport;
  }
};

template <typename T>
class Mul {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(const T* src0ptr, const T* src1ptr, T* dstptr, size_t size) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::mul(src0ptr, src1ptr, dstptr, size);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return avx2::mul(src0ptr, src1ptr, dstptr, size);
    }
#endif
    return ref::mul(src0ptr, src1ptr, dstptr, size);
  }

  static inline BTLA_CODE forward_auto(const T* src0ptr, const T* src1ptr, T* dstptr, size_t size) {
    GetCPUDevice();
    if (_cd->AVX512F()) {
      return forward<BTLA_ISA::AVX512F>(src0ptr, src1ptr, dstptr, size);
    }
    if (_cd->AVX2()) {
      return forward<BTLA_ISA::AVX2>(src0ptr, src1ptr, dstptr, size);
    }
    return forward<BTLA_ISA::NoSIMD>(src0ptr, src1ptr, dstptr, size);
  }
};

template <typename T>
class Inv {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(const T* src0ptr, T* dstptr, size_t size) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::inv(src0ptr, dstptr, size);
    }
#endif
    return ref::inv(src0ptr, dstptr, size);
  }

  static inline BTLA_CODE forward_auto(const T* src0ptr, T* dstptr, size_t size) {
    GetCPUDevice();
    if (_cd->AVX512F()) {
      return forward<BTLA_ISA::AVX512F>(src0ptr, dstptr, size);
    }
    return forward<BTLA_ISA::NoSIMD>(src0ptr, dstptr, size);
  }
};

template <typename T>
class Add {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(const T* src0ptr, const T* src1ptr, T* dstptr, size_t size) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::add(src0ptr, src1ptr, dstptr, size);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return avx2::add(src0ptr, src1ptr, dstptr, size);
    }
#endif
    return ref::add(src0ptr, src1ptr, dstptr, size);
  }

  static inline BTLA_CODE forward_auto(const T* src0ptr, const T* src1ptr, T* dstptr, size_t size) {
    GetCPUDevice();
    if (_cd->AVX512F()) {
      return forward<BTLA_ISA::AVX512F>(src0ptr, src1ptr, dstptr, size);
    }
    if (_cd->AVX2()) {
      return forward<BTLA_ISA::AVX2>(src0ptr, src1ptr, dstptr, size);
    }
    return forward<BTLA_ISA::NoSIMD>(src0ptr, src1ptr, dstptr, size);
  }
};

template <typename T_DST>
class ScaleExpAccSumFp32 {
 public:
  TLACALL BTLA_CODE forward(const float* src, const int src_step, T_DST* dst, int ld_dst, float* dst_sum,
                            const int M_offset, const int N_offset, const int M, const int N, float scale,
                            int causal_offset, void* tmpcache, size_t cachesize) {
    dst = dst + M_offset * ld_dst + N_offset;
    dst_sum = dst_sum + M_offset;
#if CompileBF16()
    if constexpr (utils::isa_base<ISA_T>::avx512_bf16 && std::is_same_v<T_DST, utils::bf16>) {
      return avx512f::avx512_bf16::scale_exp_acc_sum_fp32(src, src_step, dst, ld_dst, dst_sum, M_offset, N_offset, M, N,
                                                          scale, causal_offset, tmpcache, cachesize);
    }
#endif
    return ref::scale_exp_acc_sum_fp32(src, src_step, dst, ld_dst, dst_sum, M_offset, N_offset, M, N, scale,
                                       causal_offset, tmpcache, cachesize);
  }
};

template <typename T_SRC, typename T_DST>
class ScaleTrackMax {
 public:
  using DType = T_DST;
  using SType = T_SRC;

  TLACALL BTLA_CODE forward(const SType* src, const int src_step, DType* dst, DType* dst_max, int ld_dst,
                            const int M_offset, const int N_offset, const int M, const int N, float scale,
                            int causal_offset, float alibi_slope, float tanh_scale, int padding_type, void* tmpcache,
                            size_t cachesize) {
    dst = dst + M_offset * ld_dst + N_offset;
    dst_max = dst_max + M_offset;
    if constexpr (std::is_same_v<T_DST, float>) {
      if constexpr (std::is_same_v<T_SRC, utils::fp16>) {
        assert(padding_type != 2);
        assert(("alibi not supported!", alibi_slope == 0.f));
        assert(("tanh not supported!", tanh_scale == 0.f));
#if CompileFP16()
        if constexpr (utils::isa_base<ISA_T>::avx512_fp16) {
          return avx512f::avx512_fp16::scale_track_max_fp16_fp32(src, src_step, dst, dst_max, ld_dst, M_offset,
                                                                 N_offset, M, N, scale, causal_offset, alibi_slope,
                                                                 tanh_scale, tmpcache, cachesize);
        }
#endif
      }
      if constexpr (std::is_same_v<T_SRC, float>) {
#if CompileAVX512F()
        if constexpr (utils::isa_base<ISA_T>::avx512f) {
          return forward_avx512(src, src_step, dst, dst_max, ld_dst, M_offset, N_offset, M, N, scale, causal_offset,
                                alibi_slope, tanh_scale, padding_type, tmpcache, cachesize);
        }
#endif
#if CompileAVX2()
        if constexpr (utils::isa_base<ISA_T>::avx2) {
          return forward_avx2(src, src_step, dst, dst_max, ld_dst, M_offset, N_offset, M, N, scale, causal_offset,
                              alibi_slope, tanh_scale, padding_type, tmpcache, cachesize);
        }
#endif
      }
      if constexpr (std::is_same_v<T_SRC, int32_t>) {
        assert(padding_type != 2);
        assert(("alibi not supported!", alibi_slope == 0.f));
        assert(("tanh not supported!", tanh_scale == 0.f));
#if CompileAVX512F()
        if constexpr (utils::isa_base<ISA_T>::avx512f) {
          return avx512f::scale_track_max_int32_fp32(src, src_step, dst, dst_max, ld_dst, M_offset, N_offset, M, N,
                                                     scale, causal_offset, alibi_slope, tanh_scale, tmpcache,
                                                     cachesize);
        }
#endif
      }
    }

    return ref::scale_track_max<T_SRC, T_DST>(src, src_step, dst, dst_max, ld_dst, M_offset, N_offset, M, N, scale,
                                              causal_offset, alibi_slope, tanh_scale, padding_type, tmpcache,
                                              cachesize);
  }

  static BTLA_CODE forward_avx2(const SType* src, const int src_step, DType* dst, DType* dst_max, int ld_dst,
                                const int M_offset, const int N_offset, const int M, const int N, float scale,
                                int causal_offset, float alibi_slope, float tanh_scale, int padding_type,
                                void* tmpcache, size_t cachesize) {
#if CompileAVX2()
    if (alibi_slope == 0 && tanh_scale == 0)
      return avx2::scale_track_max_fp32_fp32<false, false>(src, src_step, dst, dst_max, ld_dst, M_offset, N_offset, M,
                                                           N, scale, causal_offset, alibi_slope, tanh_scale,
                                                           padding_type, tmpcache, cachesize);
    else if (alibi_slope == 0 && tanh_scale != 0)
      return avx2::scale_track_max_fp32_fp32<false, true>(src, src_step, dst, dst_max, ld_dst, M_offset, N_offset, M, N,
                                                          scale, causal_offset, alibi_slope, tanh_scale, padding_type,
                                                          tmpcache, cachesize);
    else if (alibi_slope != 0 && tanh_scale == 0)
      return avx2::scale_track_max_fp32_fp32<true, false>(src, src_step, dst, dst_max, ld_dst, M_offset, N_offset, M, N,
                                                          scale, causal_offset, alibi_slope, tanh_scale, padding_type,
                                                          tmpcache, cachesize);

#endif
    return BTLA_CODE::NotSupport;
  }

  static BTLA_CODE forward_avx512(const SType* src, const int src_step, DType* dst, DType* dst_max, int ld_dst,
                                  const int M_offset, const int N_offset, const int M, const int N, float scale,
                                  int causal_offset, float alibi_slope, float tanh_scale, int padding_type,
                                  void* tmpcache, size_t cachesize) {
#if CompileAVX512F()
    if (alibi_slope == 0 && tanh_scale == 0)
      return avx512f::scale_track_max_fp32_fp32<false, false>(src, src_step, dst, dst_max, ld_dst, M_offset, N_offset,
                                                              M, N, scale, causal_offset, alibi_slope, tanh_scale,
                                                              padding_type, tmpcache, cachesize);
    else if (alibi_slope == 0 && tanh_scale != 0)
      return avx512f::scale_track_max_fp32_fp32<false, true>(src, src_step, dst, dst_max, ld_dst, M_offset, N_offset, M,
                                                             N, scale, causal_offset, alibi_slope, tanh_scale,
                                                             padding_type, tmpcache, cachesize);
    else if (alibi_slope != 0 && tanh_scale == 0)
      return avx512f::scale_track_max_fp32_fp32<true, false>(src, src_step, dst, dst_max, ld_dst, M_offset, N_offset, M,
                                                             N, scale, causal_offset, alibi_slope, tanh_scale,
                                                             padding_type, tmpcache, cachesize);
#endif
    return BTLA_CODE::NotSupport;
  }
};

template <typename T_DST>
class WeightCvtBf16Ntile48 {
 public:
  using BType = T_DST;
  using SType = utils::bf16;

  TLACALL BTLA_CODE forward(const SType* B, int ldb, bool is_padded, BType* dst_ptr, int dst_step, int k_size,
                            int n_size, int k_offset, int n_offset, void* tmpcache, size_t cachesize) {
    assert(is_padded);
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f && std::is_same_v<BType, float>) {
      return avx512f::weight_cvt_bf16_fp32_n48(B, ldb, is_padded, dst_ptr, dst_step, k_size, n_size, k_offset, n_offset,
                                               tmpcache, cachesize);
    }
#endif
    return BTLA_CODE::NotSupport;
  }
};

template <typename T_DST>
class WeightCvtFp16Ntile24 {
 public:
  using BType = T_DST;
  using SType = utils::fp16;

  TLACALL BTLA_CODE forward(const SType* B, int ldb, bool is_padded, BType* dst_ptr, int dst_step, int k_size,
                            int n_size, int k_offset, int n_offset, void* tmpcache, size_t cachesize) {
    assert(is_padded);
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2 && std::is_same_v<BType, float>) {
      return avx2::weight_cvt_fp16_fp32_n24(B, ldb, is_padded, dst_ptr, dst_step, k_size, n_size, k_offset, n_offset,
                                            tmpcache, cachesize);
    }
#endif
    return BTLA_CODE::NotSupport;
  }
};

template <class SRC_T, class DST_T>
struct InplacePrecomputeMaxSoftmax {
  // nsize is the staring n-size when causal mask enabled
  // src and dst cam be on the same address if sizeof(SRC_T) >= sizeof(DST_T) and ld is correctly set
  // s_max and expsum cam be on the same address
  TLACALL BTLA_CODE forward(int m_size, int n_size, int n_pad_size, bool is_causal, SRC_T* src, DST_T* dst,
                            const SRC_T* s_max, float* expsum, int ld_src, int ld_dst) {
    if constexpr (std::is_same_v<SRC_T, float>) {
      if constexpr (std::is_same_v<DST_T, float>) {
#if CompileAVX512F()
        if constexpr (utils::isa_base<ISA_T>::avx512f) {
          return avx512f::inplace_precompute_max_softmax_fp32_fp32(m_size, n_size, n_pad_size, is_causal, src, dst,
                                                                   s_max, expsum, ld_src, ld_dst);
        }
#endif
#if CompileAVX2()
        if constexpr (utils::isa_base<ISA_T>::avx2) {
          return avx2::inplace_precompute_max_softmax_fp32_fp32(m_size, n_size, n_pad_size, is_causal, src, dst, s_max,
                                                                expsum, ld_src, ld_dst);
        }
#endif
      }
      if constexpr (std::is_same_v<DST_T, utils::fp16>) {
#if CompileFP16()
        if constexpr (utils::isa_base<ISA_T>::avx512_fp16) {
          return avx512f::avx512_fp16::inplace_precompute_max_softmax_fp32_fp16(
              m_size, n_size, n_pad_size, is_causal, src, dst, s_max, expsum, ld_src, ld_dst);
        }
#endif
      }
      if constexpr (std::is_same_v<DST_T, utils::bf16>) {
#if CompileBF16()
        if constexpr (utils::isa_base<ISA_T>::avx512_bf16) {
          return avx512f::avx512_bf16::inplace_precompute_max_softmax_fp32_bf16(
              m_size, n_size, n_pad_size, is_causal, src, dst, s_max, expsum, ld_src, ld_dst);
        }
#endif
      }
      if constexpr (std::is_same_v<DST_T, uint8_t>) {
#if CompileAVX512F()
        if constexpr (utils::isa_base<ISA_T>::avx512f) {
          return avx512f::inplace_precompute_max_softmax_fp32_u8(m_size, n_size, n_pad_size, is_causal, src, dst, s_max,
                                                                 expsum, ld_src, ld_dst);
        }
#endif
      }
    }
    assert(false);
    return BTLA_CODE::NotSupport;
  }
};
}  // namespace wrapper
}  // namespace kernel
}  // namespace bestla
