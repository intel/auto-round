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
#include <tuple>

#include "bestla.h"
#include "bestla_jit.h"
#include "bestla_utils.h"
#include "kernel_wrapper.h"

namespace bestla {
namespace epilogue {
namespace gemm {

struct ParamPcKBlockCompInt8Epilogue {
  void* scalesB;
  BTLA_DTYPE scaleBdtype;
  float* scalesA;
  // optional if A asym
  uint8_t* zpA = nullptr;
  const int32_t* reduceB = nullptr;
  // optional if B asym
  int8_t* zpB = nullptr;
  const int32_t* reduceA = nullptr;
  int K = 1;
};
template <class Fp32Epilogue>
class PcKBlockCompInt8Epilogue {
 public:
  using Fp32Param = typename Fp32Epilogue::Param;
  struct Param {
    ParamPcKBlockCompInt8Epilogue param1;
    Fp32Param param2;
  };
  using Fp32Epi = Fp32Epilogue;
  template <BTLA_ISA ISA_T>
  static BTLA_CODE forward(const int32_t* srcptr, const int cachestep, const int M_offset, const int N_offset,
                           const int M, const int N, const Param& _param, void* tmpcache, size_t cachesize) {
    BTLA_CODE ret = BTLA_CODE::NotSupport;
    auto& param1 = _param.param1;
    auto zpa_ptr = param1.zpA ? param1.zpA + M_offset : nullptr;
    auto zpb_ptr = param1.zpB ? param1.zpB + N_offset : nullptr;
    if (zpa_ptr || zpb_ptr) {
      ret = kernel::wrapper::RemoveZeroPointBias::template forward<ISA_T>(
          const_cast<int32_t*>(srcptr), cachestep, M, N, zpa_ptr, zpb_ptr, 1, param1.K, param1.reduceA + M_offset,
          param1.reduceB + N_offset);
      assert(ret == BTLA_CODE::Success);
    }
    auto tmpfp32ptr = reinterpret_cast<float*>(const_cast<int32_t*>(srcptr));
    if (param1.scaleBdtype == BTLA_DTYPE::BF16) {
      auto sbptr = reinterpret_cast<utils::bf16*>(param1.scalesB) + N_offset;
      ret = kernel::wrapper::DequanS32Fp32::template forward<ISA_T>(srcptr, cachestep, tmpfp32ptr, cachestep, M, N,
                                                                    param1.scalesA + M_offset, 1, sbptr);
    } else if (param1.scaleBdtype == BTLA_DTYPE::F16) {
      auto sbptr = reinterpret_cast<utils::fp16*>(param1.scalesB) + N_offset;
      ret = kernel::wrapper::DequanS32Fp32::template forward<ISA_T>(srcptr, cachestep, tmpfp32ptr, cachestep, M, N,
                                                                    param1.scalesA + M_offset, 1, sbptr);
    } else if (param1.scaleBdtype == BTLA_DTYPE::F32) {
      auto sbptr = reinterpret_cast<float*>(param1.scalesB) + N_offset;
      ret = kernel::wrapper::DequanS32Fp32::template forward<ISA_T>(srcptr, cachestep, tmpfp32ptr, cachestep, M, N,
                                                                    param1.scalesA + M_offset, 1, sbptr);
    }
    assert(ret == BTLA_CODE::Success);
    ret = Fp32Epilogue::template forward<ISA_T>(tmpfp32ptr, cachestep, M_offset, N_offset, M, N, _param.param2,
                                                tmpcache, cachesize);
    return ret;
  }
};

template <typename DT>
struct ParamAccumulatorWriteBack {
  DT* C;
  int ldc;
  void* elt_const_v;
};

template <typename _SRC_T, typename _DST_T>
class AccumulatorWriteBack {
 public:
  using SType = _SRC_T;
  using DType = _DST_T;
  using Param = ParamAccumulatorWriteBack<DType>;
  using PcCompInt8Epi = bestla::epilogue::gemm::PcKBlockCompInt8Epilogue<AccumulatorWriteBack<_SRC_T, _DST_T>>;

  template <BTLA_ISA ISA_T>
  static BTLA_CODE forward(const _SRC_T* cacheptr, const int cachestep, const int M_offset, const int N_offset,
                           const int M, const int N, const Param& _param, void* tmpcache, size_t cachesize) {
    auto COffset = M_offset * _param.ldc + N_offset;
    auto cptr = _param.C + COffset;
    if constexpr (std::is_same_v<_SRC_T, DType>) {
      if (cacheptr == cptr) {
        return BTLA_CODE::Success;
      }
    }

    return kernel::wrapper::Memcpy2D::template forward<ISA_T, SType, DType>(cacheptr, cptr, M, N, cachestep, _param.ldc,
                                                                            _param.elt_const_v);
  }
};

template <typename _SRC_T, typename _DST_T, BTLA_ELTWISEOP _OP>
class CustomAccumulatorWriteBackWithEltop {
 public:
  using PcCompInt8Epi =
      bestla::epilogue::gemm::PcKBlockCompInt8Epilogue<CustomAccumulatorWriteBackWithEltop<_SRC_T, _DST_T, _OP>>;
  using Param = ParamAccumulatorWriteBack<_DST_T>;
  template <BTLA_ISA ISA_T>
  static BTLA_CODE forward(const _SRC_T* cacheptr, const int cachestep, const int M_offset, const int N_offset,
                           const int M, const int N, const Param& _param, void* tmpcache, size_t cachesize) {
    auto COffset = M_offset * _param.ldc + N_offset;
    auto cptr = _param.C + COffset;
    if constexpr (std::is_same<_SRC_T, float>::value && std::is_same<_DST_T, float>::value) {
      return kernel::wrapper::Memcpy2D::template forward1<ISA_T, float, float, _OP>(cacheptr, cptr, M, N, cachestep,
                                                                                    _param.ldc, _param.elt_const_v);
    } else {
      assert(false);
    }
  }
};
using AccumulatorWriteBackFp32 = AccumulatorWriteBack<float, float>;
using AccumulatorWriteBackInt32 = AccumulatorWriteBack<int, int>;
using AccumulatorWriteBackBf16 = AccumulatorWriteBack<utils::bf16, utils::bf16>;
using AccumulatorWriteBackFp16 = AccumulatorWriteBack<utils::fp16, utils::fp16>;
using AccumulatorWriteBackBf16Fp32 = AccumulatorWriteBack<utils::bf16, float>;
using AccumulatorWriteBackFp16Fp32 = AccumulatorWriteBack<utils::fp16, float>;
using AccumulatorWriteBackFp32Bf16 = AccumulatorWriteBack<float, utils::bf16>;

using AccumulatorWriteBackWithGeluFp32 = CustomAccumulatorWriteBackWithEltop<float, float, BTLA_ELTWISEOP::GELU>;

using AccumulatorWriteBackWithSwishFp32 = CustomAccumulatorWriteBackWithEltop<float, float, BTLA_ELTWISEOP::SWISH>;

template <typename DT>
struct ParamAlphaBetaProcess {
  DT *C, *D;
  int ldc, ldd;
  float alpha, beta;
};
class AlphaBetaProcessFp32 {
 public:
  using Param = ParamAlphaBetaProcess<float>;

  template <BTLA_ISA ISA_T>
  static BTLA_CODE forward(const float* cacheptr, const int cachestep, const int M_offset, const int N_offset,
                           const int M, const int N, const Param& _param, void* tmpcache, size_t cachesize) {
    auto DOffset = M_offset * _param.ldd + N_offset;
    auto COffset = M_offset * _param.ldc + N_offset;
    auto cptr = _param.C + COffset;
    auto dptr = _param.D + DOffset;
    return kernel::wrapper::AlphaBetaF32F32::template forward<ISA_T>(_param.alpha, cacheptr, cachestep, _param.beta,
                                                                     dptr, _param.ldd, cptr, _param.ldc, M, N);
  }
};

struct ParamDequantInt32ToFp32 {
  float* C;
  int ldc;
  int ldsa;
  float* scalesA;
  float* scalesB;
};
class DequantInt32ToFp32 {
 public:
  using Param = ParamDequantInt32ToFp32;
  template <BTLA_ISA ISA_T>
  static BTLA_CODE forward(const int32_t* cacheptr, const int cachestep, const int M_offset, const int N_offset,
                           const int M, const int N, const Param& _param, void* tmpcache, size_t cachesize) {
    auto COffset = M_offset * _param.ldc + N_offset;
    auto cptr = _param.C + COffset;
    return kernel::wrapper::DequanS32Fp32::template forward<ISA_T>(cacheptr, cachestep, cptr, _param.ldc, M, N,
                                                                   _param.scalesA + M_offset * _param.ldsa, _param.ldsa,
                                                                   _param.scalesB + N_offset);
  }
};

struct ParamZpDequantInt32ToFp32 {
  // necessary
  float* C;
  int ldc;
  int ldsa;
  float* scalesA;
  float* scalesB;
  // optional if A asym
  uint8_t* zpA = nullptr;
  int32_t* reduceB = nullptr;
  // optional if B asym
  int8_t* zpB = nullptr;
  int32_t* reduceA = nullptr;
  int K = 1;
};
class ZpDequantInt32ToFp32 {
 public:
  using Param = ParamZpDequantInt32ToFp32;
  template <BTLA_ISA ISA_T>
  static BTLA_CODE forward(const int32_t* cacheptr, const int cachestep, const int M_offset, const int N_offset,
                           const int M, const int N, const Param& _param, void* tmpcache, size_t cachesize) {
    auto COffset = M_offset * _param.ldc + N_offset;
    auto cptr = _param.C + COffset;
    auto zpa_ptr = _param.zpA ? _param.zpA + M_offset : nullptr;
    auto zpb_ptr = _param.zpB ? _param.zpB + N_offset : nullptr;
    auto ret = BTLA_CODE::Success;
    if (zpa_ptr || zpb_ptr) {
      ret = kernel::wrapper::RemoveZeroPointBias::template forward<ISA_T>(
          const_cast<int32_t*>(cacheptr), cachestep, M, N, zpa_ptr, zpb_ptr, _param.ldsa, _param.K,
          _param.reduceA + M_offset * _param.ldsa, _param.reduceB + N_offset);
      assert(ret == BTLA_CODE::Success);
    }
    ret = kernel::wrapper::DequanS32Fp32::template forward<ISA_T>(cacheptr, cachestep, cptr, _param.ldc, M, N,
                                                                  _param.scalesA + M_offset * _param.ldsa, _param.ldsa,
                                                                  _param.scalesB + N_offset);
    if (ret != BTLA_CODE::Success) {
      return ret;
    }
    return ret;
  }
};

struct ParamAlphaBetaProcessS32U8 {
  uint8_t* C;
  int ldc;
  float alpha;
  float scaleAcc, scaleC;
  int zpC;
};
class AlphaBetaProcessS32U8 {
 public:
  using Param = ParamAlphaBetaProcessS32U8;
  template <BTLA_ISA ISA_T>
  static BTLA_CODE forward(const int32_t* cacheptr, const int cachestep, const int M_offset, const int N_offset,
                           const int M, const int N, const Param& _param, void* tmpcache, size_t cachesize) {
    auto COffset = M_offset * _param.ldc + N_offset;
    auto cptr = _param.C + COffset;
    return kernel::wrapper::QuanOutS32U32::template forward<ISA_T>(_param.alpha, cacheptr, cachestep, cptr, _param.ldc,
                                                                   M, N, _param.scaleAcc, _param.scaleC, _param.zpC);
  }
};

}  // namespace gemm
}  // namespace epilogue
}  // namespace bestla
