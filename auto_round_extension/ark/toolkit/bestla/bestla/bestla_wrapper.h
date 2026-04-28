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
#include <thread>

#include "bestla_epilogue.h"
#include "bestla_gemm.h"
#include "bestla_gemm_kblock.h"
#include "bestla_prologue_a.h"
#include "bestla_prologue_b.h"
#include "bestla_utils.h"

namespace bestla {
namespace wrapper {
namespace gemv_nbits {
class S7 {
 public:
  static int constexpr NBits = 7;
  template <typename ScaleT>
  static inline utils::GemvParamB<ScaleT> createB(storage::gemm::StorageWeightKBlockNInteger* packedW) {
    auto isasym = packedW->IsAsym();
    auto bzptr = packedW->template ZPtr<int8_t>();
    int ld_scaleb = packedW->CStep();
    auto bwptr = packedW->template WPtr<uint8_t>();
    auto bit2_offset = packedW->mNPad * packedW->mKPad / 2;
    auto bit1_offset = packedW->mNPad * packedW->mKPad / 2 + packedW->mNPad * packedW->mKPad / 4;
    utils::GemvParamB<ScaleT> paramB{
        bwptr, bwptr + bit2_offset, bwptr + bit1_offset, packedW->template SPtr<ScaleT>(), isasym ? bzptr : nullptr,
        NBits, ld_scaleb,           packedW->mKPad};
    return paramB;
  }
  template <typename ScaleT>
  static void updateBNStep(utils::GemvParamB<ScaleT>& paramB, int n_offset) {
    paramB.b4ptr += n_offset * paramB.kpad / 2;
    paramB.b2ptr += n_offset * paramB.kpad / 4;
    paramB.b1ptr += n_offset * paramB.kpad / 8;
    paramB.sptr += n_offset;
    if (paramB.zpptr) {
      paramB.zpptr += n_offset;
    }
  }
};

class S6 {
 public:
  static int constexpr NBits = 6;
  template <typename ScaleT>
  static inline utils::GemvParamB<ScaleT> createB(storage::gemm::StorageWeightKBlockNInteger* packedW) {
    auto isasym = packedW->IsAsym();
    auto bzptr = packedW->template ZPtr<int8_t>();
    int ld_scaleb = packedW->CStep();
    auto bwptr = packedW->template WPtr<uint8_t>();
    auto bit2_offset = packedW->mNPad * packedW->mKPad / 2;
    utils::GemvParamB<ScaleT> paramB{
        bwptr, bwptr + bit2_offset, nullptr,       packedW->template SPtr<ScaleT>(), isasym ? bzptr : nullptr,
        NBits, ld_scaleb,           packedW->mKPad};
    return paramB;
  }
  template <typename ScaleT>
  static void updateBNStep(utils::GemvParamB<ScaleT>& paramB, int n_offset) {
    paramB.b4ptr += n_offset * paramB.kpad / 2;
    paramB.b2ptr += n_offset * paramB.kpad / 4;
    paramB.sptr += n_offset;
    if (paramB.zpptr) {
      paramB.zpptr += n_offset;
    }
  }
};

class S5 {
 public:
  static int constexpr NBits = 5;
  template <typename ScaleT>
  static inline utils::GemvParamB<ScaleT> createB(storage::gemm::StorageWeightKBlockNInteger* packedW) {
    auto isasym = packedW->IsAsym();
    auto bzptr = packedW->template ZPtr<int8_t>();
    int ld_scaleb = packedW->CStep();
    auto bwptr = packedW->template WPtr<uint8_t>();
    auto bit1_offset = packedW->mNPad * packedW->mKPad / 2;
    utils::GemvParamB<ScaleT> paramB{
        bwptr, nullptr,   bwptr + bit1_offset, packedW->template SPtr<ScaleT>(), isasym ? bzptr : nullptr,
        NBits, ld_scaleb, packedW->mKPad};
    return paramB;
  }
  template <typename ScaleT>
  static void updateBNStep(utils::GemvParamB<ScaleT>& paramB, int n_offset) {
    paramB.b4ptr += n_offset * paramB.kpad / 2;
    paramB.b1ptr += n_offset * paramB.kpad / 8;
    paramB.sptr += n_offset;
    if (paramB.zpptr) {
      paramB.zpptr += n_offset;
    }
  }
};

class S4 {
 public:
  static int constexpr NBits = 4;
  template <typename ScaleT>
  static inline utils::GemvParamB<ScaleT> createB(storage::gemm::StorageWeightKBlockNInteger* packedW) {
    auto isasym = packedW->IsAsym();
    auto bzptr = packedW->template ZPtr<int8_t>();
    int ld_scaleb = packedW->CStep();
    utils::GemvParamB<ScaleT> paramB{
        packedW->template WPtr<uint8_t>(), nullptr, nullptr,   packedW->template SPtr<ScaleT>(),
        isasym ? bzptr : nullptr,          NBits,   ld_scaleb, packedW->mKPad};
    return paramB;
  }
  template <typename ScaleT>
  static void updateBNStep(utils::GemvParamB<ScaleT>& paramB, int n_offset) {
    paramB.b4ptr += n_offset * paramB.kpad / 2;
    paramB.sptr += n_offset;
    if (paramB.zpptr) {
      paramB.zpptr += n_offset;
    }
  }
};

class S3 {
 public:
  static int constexpr NBits = 3;
  template <typename ScaleT>
  static inline utils::GemvParamB<ScaleT> createB(storage::gemm::StorageWeightKBlockNInteger* packedW) {
    auto isasym = packedW->IsAsym();
    auto bzptr = packedW->template ZPtr<int8_t>();
    int ld_scaleb = packedW->CStep();
    auto bwptr = packedW->template WPtr<uint8_t>();
    auto bit1_offset = packedW->mNPad * packedW->mKPad / 4;
    utils::GemvParamB<ScaleT> paramB{
        nullptr, bwptr,     bwptr + bit1_offset, packedW->template SPtr<ScaleT>(), isasym ? bzptr : nullptr,
        NBits,   ld_scaleb, packedW->mKPad};
    return paramB;
  }
  template <typename ScaleT>
  static void updateBNStep(utils::GemvParamB<ScaleT>& paramB, int n_offset) {
    paramB.b2ptr += n_offset * paramB.kpad / 4;
    paramB.b1ptr += n_offset * paramB.kpad / 8;
    paramB.sptr += n_offset;
    if (paramB.zpptr) {
      paramB.zpptr += n_offset;
    }
  }
};

class S2 {
 public:
  static int constexpr NBits = 2;
  template <typename ScaleT>
  static inline utils::GemvParamB<ScaleT> createB(storage::gemm::StorageWeightKBlockNInteger* packedW) {
    auto isasym = packedW->IsAsym();
    auto bzptr = packedW->template ZPtr<int8_t>();
    int ld_scaleb = packedW->CStep();
    auto bwptr = packedW->template WPtr<uint8_t>();
    utils::GemvParamB<ScaleT> paramB{
        nullptr, bwptr,     nullptr,       packedW->template SPtr<ScaleT>(), isasym ? bzptr : nullptr,
        NBits,   ld_scaleb, packedW->mKPad};
    return paramB;
  }
  template <typename ScaleT>
  static void updateBNStep(utils::GemvParamB<ScaleT>& paramB, int n_offset) {
    paramB.b2ptr += n_offset * paramB.kpad / 4;
    paramB.sptr += n_offset;
    if (paramB.zpptr) {
      paramB.zpptr += n_offset;
    }
  }
};

class S1 {
 public:
  static int constexpr NBits = 1;
  template <typename ScaleT>
  static inline utils::GemvParamB<ScaleT> createB(storage::gemm::StorageWeightKBlockNInteger* packedW) {
    auto isasym = packedW->IsAsym();
    auto bzptr = packedW->template ZPtr<int8_t>();
    int ld_scaleb = packedW->CStep();
    auto bwptr = packedW->template WPtr<uint8_t>();
    utils::GemvParamB<ScaleT> paramB{
        nullptr, nullptr,   bwptr,         packedW->template SPtr<ScaleT>(), isasym ? bzptr : nullptr,
        NBits,   ld_scaleb, packedW->mKPad};
    return paramB;
  }
  template <typename ScaleT>
  static void updateBNStep(utils::GemvParamB<ScaleT>& paramB, int n_offset) {
    paramB.b1ptr += n_offset * paramB.kpad / 8;
    paramB.sptr += n_offset;
    if (paramB.zpptr) {
      paramB.zpptr += n_offset;
    }
  }
};

class NBitsHelper {
 public:
  template <typename ScaleT>
  static inline utils::GemvParamB<ScaleT> createB(storage::gemm::StorageWeightKBlockNInteger* packedW) {
    if (packedW->mDType == BTLA_DTYPE::S4_CLIP) {
      return S4::createB<ScaleT>(packedW);
    }
    if (packedW->mDType == BTLA_DTYPE::S3_CLIP) {
      return S3::createB<ScaleT>(packedW);
    }
    if (packedW->mDType == BTLA_DTYPE::S5_CLIP) {
      return S5::createB<ScaleT>(packedW);
    }
    if (packedW->mDType == BTLA_DTYPE::S2_CLIP) {
      return S2::createB<ScaleT>(packedW);
    }
    if (packedW->mDType == BTLA_DTYPE::S6_CLIP) {
      return S6::createB<ScaleT>(packedW);
    }
    if (packedW->mDType == BTLA_DTYPE::S7_CLIP) {
      return S7::createB<ScaleT>(packedW);
    }
    if (packedW->mDType == BTLA_DTYPE::S1_CLIP) {
      return S1::createB<ScaleT>(packedW);
    }
    assert(0);
    return utils::GemvParamB<ScaleT>();
  }
  template <typename ScaleT>
  static void updateBNStep(utils::GemvParamB<ScaleT>& paramB, int n_offset) {
    if (paramB.nbits == 4) {
      return S4::updateBNStep<ScaleT>(paramB, n_offset);
    }
    if (paramB.nbits == 3) {
      return S3::updateBNStep<ScaleT>(paramB, n_offset);
    }
    if (paramB.nbits == 5) {
      return S5::updateBNStep<ScaleT>(paramB, n_offset);
    }
    if (paramB.nbits == 2) {
      return S2::updateBNStep<ScaleT>(paramB, n_offset);
    }
    if (paramB.nbits == 6) {
      return S6::updateBNStep<ScaleT>(paramB, n_offset);
    }
    if (paramB.nbits == 7) {
      return S7::updateBNStep<ScaleT>(paramB, n_offset);
    }
    if (paramB.nbits == 1) {
      return S1::updateBNStep<ScaleT>(paramB, n_offset);
    }
    assert(0);
  }
};

class GemvWeight {
 public:
  template <typename ScaleT>
  static inline utils::GemvParamB<ScaleT> createB(const storage::gemm::StorageWeightNInt* packedW) {
    auto bzptr = packedW->corr_.zptr();
    int ld_scaleb = packedW->corr_.step_;
    auto bwptr = packedW->buf_.buf_ptr_;
    auto nbits = (int)utils::bestla_dtype_bits(packedW->info_.dtype_);
    utils::GemvParamB<ScaleT> paramB{nullptr, nullptr, nullptr,   (ScaleT*)packedW->corr_.scale_buf_.buf_ptr_,
                                     bzptr,   nbits,   ld_scaleb, packedW->info_.kpad_};
    auto bit2_offset = packedW->info_.npad_ * packedW->info_.kpad_ / 2;
    auto bit1_offset =
        packedW->info_.npad_ * packedW->info_.kpad_ / 2 + packedW->info_.npad_ * packedW->info_.kpad_ / 4;
    if (nbits & 0x4) {
      paramB.b4ptr = (uint8_t*)bwptr;
      bwptr += packedW->info_.npad_ * packedW->info_.kpad_ / 2;
    }
    if (nbits & 0x2) {
      paramB.b2ptr = (uint8_t*)bwptr;
      bwptr += packedW->info_.npad_ * packedW->info_.kpad_ / 4;
    }
    if (nbits & 0x1) {
      paramB.b1ptr = (uint8_t*)bwptr;
    }
    return paramB;
  }
  template <typename ScaleT>
  static void updateBNStep(utils::GemvParamB<ScaleT>& paramB, int n_offset) {
    if (paramB.b4ptr) paramB.b4ptr += n_offset * paramB.kpad / 2;
    if (paramB.b2ptr) paramB.b2ptr += n_offset * paramB.kpad / 4;
    if (paramB.b1ptr) paramB.b1ptr += n_offset * paramB.kpad / 8;
    paramB.sptr += n_offset;
    if (paramB.zpptr) paramB.zpptr += n_offset;
  }
};

}  // namespace gemv_nbits

namespace gemm {
template <class _GemmCore_T, template <class _T> class _PrologueA_T, template <class _T> class _PrologueB_T,
          class _Epilogue_T>
class LauncherBase {
 public:
  using GemmCore = _GemmCore_T;
  static constexpr BTLA_ISA ISA = GemmCore::ISA;
  using PrologueA = _PrologueA_T<GemmCore>;
  using PrologueB = _PrologueB_T<GemmCore>;
  using Epilogue = _Epilogue_T;
  using AType = typename GemmCore::AType;
  using AParam = typename PrologueA::Param;
  using BType = typename GemmCore::BType;
  using BParam = typename PrologueB::Param;
  using CType = typename GemmCore::CType;
  using EpiParam = typename Epilogue::Param;
  struct Param {
    const utils::GemmProblem problem;
    const AParam paramA;
    const BParam paramB;
    const EpiParam paramC;
  };

  class GEMVWrapper {
   public:
    static constexpr bool support() {
      if constexpr (!std::is_same_v<PrologueB, prologue_b::gemm::WeightKBlockNInteger<_GemmCore_T>>) {
        return false;
      }
      if constexpr (!std::is_same_v<PrologueA, prologue_a::gemm::ShuffleActivationKBlockBaseF32<_GemmCore_T>> &&
                    !std::is_same_v<PrologueA, prologue_a::gemm::ActivationKBlockBaseF32<_GemmCore_T>> &&
                    !std::is_same_v<PrologueA, prologue_a::gemm::ActivationF32KBlockQuantize<_GemmCore_T>> &&
                    !std::is_same_v<PrologueA, prologue_a::gemm::ShuffleActivationKBlockQuantizeF32<_GemmCore_T>> &&
                    !std::is_same_v<PrologueA, prologue_a::gemm::ActivationBase<_GemmCore_T>>) {
        return false;
      }

      if constexpr (GemmCore::ISA == BTLA_ISA::AVX2) {
#if CompileAVX2()
        if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_FP32) {
          static_assert(GemmCore::PACK_ROW == 1);
          return true;
        }
        if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_INT8_US_INT32) {
          static_assert(GemmCore::PACK_ROW == 4);
          return true;
        }
#endif
      }
      if constexpr (GemmCore::ISA == BTLA_ISA::AVX512_VNNI || GemmCore::ISA == BTLA_ISA::AMX_INT8) {
#if CompileAVX512VNNI()
        if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_INT8_US_INT32) {
          static_assert(GemmCore::PACK_ROW == 4);
          return true;
        }
#endif
      }
      if constexpr (GemmCore::ISA == BTLA_ISA::AVX_VNNI) {
#if CompileAVXVNNI()
        if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_INT8_US_INT32) {
          static_assert(GemmCore::PACK_ROW == 4);
          return true;
        }
#endif
      }
      if constexpr (GemmCore::ISA == BTLA_ISA::AVX512F) {
#if CompileAVX512F()
        if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_FP32) {
          static_assert(GemmCore::PACK_ROW == 1);
          return true;
        }
#endif
      }
      if constexpr (GemmCore::ISA == BTLA_ISA::AVX512BW) {
#if CompileAVX512F()
        if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_INT8_US_INT32) {
          static_assert(GemmCore::PACK_ROW == 4);
          return true;
        }
#endif
      }
      return false;
    }
    static bool constexpr Reg32 = GemmCore::ISA >= BTLA_ISA::AVX512F;
    static int constexpr MaxGemvM = Reg32 ? 4 : 4;
    static bool implemented(const Param& _param) {
      bool impl = true;
      impl &= _param.paramB.packedW->mDType == BTLA_DTYPE::S4_CLIP ||
              _param.paramB.packedW->mDType == BTLA_DTYPE::S3_CLIP ||
              _param.paramB.packedW->mDType == BTLA_DTYPE::S7_CLIP ||
              _param.paramB.packedW->mDType == BTLA_DTYPE::S6_CLIP ||
              _param.paramB.packedW->mDType == BTLA_DTYPE::S5_CLIP ||
              _param.paramB.packedW->mDType == BTLA_DTYPE::S1_CLIP ||
              _param.paramB.packedW->mDType == BTLA_DTYPE::S2_CLIP;
      if constexpr (support()) {
        impl &= _param.paramB.packedW->mCorrection.mScaT == BTLA_DTYPE::F32 ||
                _param.paramB.packedW->mCorrection.mScaT == BTLA_DTYPE::F16 ||
                _param.paramB.packedW->mCorrection.mScaT == BTLA_DTYPE::BF16;
      }

      impl &= _param.problem.dims[1] <= MaxGemvM;
      return impl;
    }

    template <typename ScaleT, int MTILE>
    static void gemv_kblock(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
      if constexpr (support()) {
        auto constexpr TmpSize = 16 * 1024LL;
        auto constexpr CSize = 8 * 1024LL;
        auto StackTmp_ = alloca(TmpSize + CSize);
        auto StackTmp = utils::cpu_pointer_align<void>(StackTmp_);
        auto tmpc_ptr = reinterpret_cast<float*>((char*)StackTmp + TmpSize);
        static_assert(CSize >= (MTILE * GemmCore::NTILE * sizeof(float)));
        utils::GemvParamB<ScaleT> paramB = gemv_nbits::NBitsHelper::template createB<ScaleT>(_param.paramB.packedW);
        int m = _param.problem.dims[1];
        int n = _param.problem.dims[2];
        int k = _param.problem.dims[3];
        int kblocksize = _param.problem.dims[4];
        gemv_nbits::NBitsHelper::template updateBNStep<ScaleT>(paramB, _config.loc[1]);
        int size_padded = utils::padto_le(_config.size[1], GemmCore::NTILE);
        int in = 0;
        for (; in < size_padded; in += GemmCore::NTILE) {
          if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_INT8_US_INT32) {
            utils::GemvParamA paramA{
                _param.paramA.quan->template APtr<uint8_t>(), _param.paramA.quan->template SPtr<float>(),
                _param.paramA.quan->template ZPtr<uint8_t>(), _param.paramA.quan->mKPad, _param.paramA.quan->CStep()};
            kernel::wrapper::GEMVWoqNBits::forward_u8s8_fp32<ISA, ScaleT, GemmCore::NTILE, MTILE>(
                paramA, paramB, tmpc_ptr, GemmCore::NTILE, k, kblocksize, StackTmp, TmpSize);
            Epilogue::Fp32Epi::template forward<ISA>(tmpc_ptr, GemmCore::NTILE, 0, _config.loc[1] + in, MTILE,
                                                     GemmCore::NTILE, _param.paramC.param2, StackTmp, TmpSize);
          } else {
            const float* Aptr = _param.paramA.A;
            if constexpr (std::is_same_v<PrologueA, prologue_a::gemm::ShuffleActivationKBlockBaseF32<_GemmCore_T>>) {
              if (_param.paramA.reordered && _param.paramA.reordered->template APtr<float>()) {
                Aptr = _param.paramA.reordered->template APtr<float>();
              }
            }
            kernel::wrapper::GEMVWoqNBits::forward_fp32_fp32<ISA, ScaleT, GemmCore::NTILE, MTILE>(
                Aptr, _param.paramA.lda, paramB, tmpc_ptr, GemmCore::NTILE, k, kblocksize, StackTmp, TmpSize);
            Epilogue::template forward<ISA>(tmpc_ptr, GemmCore::NTILE, 0, _config.loc[1] + in, MTILE, GemmCore::NTILE,
                                            _param.paramC, StackTmp, TmpSize);
          }
          gemv_nbits::NBitsHelper::template updateBNStep<ScaleT>(paramB, GemmCore::NTILE);
        }
        if (size_padded != _config.size[1]) {
          if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_INT8_US_INT32) {
            utils::GemvParamA paramA{
                _param.paramA.quan->template APtr<uint8_t>(), _param.paramA.quan->template SPtr<float>(),
                _param.paramA.quan->template ZPtr<uint8_t>(), _param.paramA.quan->mKPad, _param.paramA.quan->CStep()};
            kernel::wrapper::GEMVWoqNBits::forward_u8s8_fp32<ISA, ScaleT, GemmCore::NTILE, MTILE>(
                paramA, paramB, tmpc_ptr, GemmCore::NTILE, k, kblocksize, StackTmp, TmpSize);
            Epilogue::Fp32Epi::template forward<ISA>(tmpc_ptr, GemmCore::NTILE, 0, _config.loc[1] + in, MTILE,
                                                     (_config.size[1] - in), _param.paramC.param2, StackTmp, TmpSize);
          } else {
            const float* Aptr = _param.paramA.A;
            if constexpr (std::is_same_v<PrologueA, prologue_a::gemm::ShuffleActivationKBlockBaseF32<_GemmCore_T>>) {
              if (_param.paramA.reordered && _param.paramA.reordered->template APtr<float>()) {
                Aptr = _param.paramA.reordered->template APtr<float>();
              }
            }
            kernel::wrapper::GEMVWoqNBits::forward_fp32_fp32<ISA, ScaleT, GemmCore::NTILE, MTILE>(
                Aptr, _param.paramA.lda, paramB, tmpc_ptr, GemmCore::NTILE, k, kblocksize, StackTmp, TmpSize);
            Epilogue::template forward<ISA>(tmpc_ptr, GemmCore::NTILE, 0, _config.loc[1] + in, MTILE,
                                            (_config.size[1] - in), _param.paramC, StackTmp, TmpSize);
          }
        }
      }
    }

    static void gemv(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
      if constexpr (support()) {
        assert(_param.problem.dims[4] > 0);
        auto& m = _param.problem.dims[1];
        if (_param.paramB.packedW->SDtype() == BTLA_DTYPE::F32) {
          if (m == 1) gemv_kblock<float, 1>(_param, _config);
          if (m == 2) gemv_kblock<float, 2>(_param, _config);
          if (m == 3) gemv_kblock<float, 3>(_param, _config);
          if (m == 4) gemv_kblock<float, 4>(_param, _config);
          if constexpr (Reg32) {
            if (m == 5) gemv_kblock<float, 5>(_param, _config);
            if (m == 6) gemv_kblock<float, 6>(_param, _config);
            if (m == 7) gemv_kblock<float, 7>(_param, _config);
            if (m == 8) gemv_kblock<float, 8>(_param, _config);
          }
        } else if (_param.paramB.packedW->SDtype() == BTLA_DTYPE::BF16) {
          if (m == 1) gemv_kblock<utils::bf16, 1>(_param, _config);
          if (m == 2) gemv_kblock<utils::bf16, 2>(_param, _config);
          if (m == 3) gemv_kblock<utils::bf16, 3>(_param, _config);
          if (m == 4) gemv_kblock<utils::bf16, 4>(_param, _config);
          if constexpr (Reg32) {
            if (m == 5) gemv_kblock<utils::bf16, 5>(_param, _config);
            if (m == 6) gemv_kblock<utils::bf16, 6>(_param, _config);
            if (m == 7) gemv_kblock<utils::bf16, 7>(_param, _config);
            if (m == 8) gemv_kblock<utils::bf16, 8>(_param, _config);
          }
        } else if (_param.paramB.packedW->SDtype() == BTLA_DTYPE::F16) {
          if (m == 1) gemv_kblock<utils::fp16, 1>(_param, _config);
          if (m == 2) gemv_kblock<utils::fp16, 2>(_param, _config);
          if (m == 3) gemv_kblock<utils::fp16, 3>(_param, _config);
          if (m == 4) gemv_kblock<utils::fp16, 4>(_param, _config);
          if constexpr (Reg32) {
            if (m == 5) gemv_kblock<utils::fp16, 5>(_param, _config);
            if (m == 6) gemv_kblock<utils::fp16, 6>(_param, _config);
            if (m == 7) gemv_kblock<utils::fp16, 7>(_param, _config);
            if (m == 8) gemv_kblock<utils::fp16, 8>(_param, _config);
          }
        }
      }
    }
  };

  static void run(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
    if (GEMVWrapper::support() && GEMVWrapper::implemented(_param)) {
      GEMVWrapper::gemv(_param, _config);
    } else {
      gemm(_param, _config);
    }
    bestla::kernel::wrapper::ZeroReg::forward();
  }

 protected:
  static void gemm(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
    GemmCore::configure(_config.size[0], _config.size[1], _param.problem.dims[3]);
    auto StackTmp = alloca(_config.stacksize);
    auto tmpB = reinterpret_cast<BType*>(StackTmp);
    tmpB = utils::cpu_pointer_align(tmpB);
    auto tmpA = reinterpret_cast<AType*>(tmpB + static_cast<size_t>(_config.block[1]) * _config.block[2]);
    tmpA = utils::cpu_pointer_align(tmpA);
    auto tmpC = reinterpret_cast<CType*>(tmpA + static_cast<size_t>(GemmCore::MTILE) * _config.block[2]);
    tmpC = utils::cpu_pointer_align(tmpC);
    auto tmpCache = (void*)(tmpC + static_cast<size_t>(_config.block[0]) * _config.block[1]);
    tmpCache = utils::cpu_pointer_align(tmpCache);
    for (int itern = 0; itern < _config.size[1]; itern += _config.block[1]) {
      int n_remain = utils::remainsize(itern, _config.size[1], _config.block[1]);
      for (int iterm = 0; iterm < _config.size[0]; iterm += _config.block[0]) {
        int m_remain = utils::remainsize(iterm, _config.size[0], _config.block[0]);
        run_block(_param, _config, iterm, itern, m_remain, n_remain, tmpA, tmpB, tmpC, tmpCache);
      }
    }
  }

  static void run_block(const Param& _param, const parallel::gemm::ThreadProblemBase& _config, int blk_m, int blk_n,
                        int blk_msize, int blk_nsize, AType* tmpA, BType* tmpB, CType* tmpC, void* tmpcache) {
    int n_padded = utils::padto(blk_nsize, GemmCore::NTILE);
    auto& K = _param.problem.dims[3];
    for (int iterk = 0; iterk < _param.problem.dims[3]; iterk += _config.block[2]) {
      int k_remain = utils::remainsize(iterk, K, _config.block[2]);
      int k_padded = utils::padto(k_remain, GemmCore::KTILE);
      int k_paddedle = utils::padto_le(k_remain, GemmCore::KTILE);
      auto bptr_cache = tmpB;
      int bcache_step = 0;
      PrologueB::template getWeight<ISA>(&bptr_cache, &bcache_step, k_padded, n_padded, iterk, _config.loc[1] + blk_n,
                                         _param.paramB, tmpcache, _config.tmpcachesize);
      int bcache_stride = bcache_step * sizeof(BType);
      for (int i = 0; i < blk_msize; i += GemmCore::MTILE) {
        int m_remain = utils::remainsize(i, blk_msize, GemmCore::MTILE);
        auto cptr_cache = tmpC + i * _config.block[1];
        int ccache_stride = _config.block[1] * sizeof(CType);
        if (k_paddedle) {
          AType* aptr_cache = tmpA;
          int acache_step = 0;
          PrologueA::template getActivation<ISA>(&aptr_cache, &acache_step, _param.paramA, m_remain, k_paddedle,
                                                 (blk_m + i + _config.loc[0]), iterk, tmpcache, _config.tmpcachesize);
          GemmCore::forward(aptr_cache, bptr_cache, cptr_cache, m_remain, n_padded, k_paddedle,
                            acache_step * sizeof(AType), bcache_stride, ccache_stride, iterk, tmpcache,
                            _config.tmpcachesize);
        }
        int k_tail = k_remain - k_paddedle;
        if (k_tail) {
          AType* aptr_cache = tmpA;
          int acache_step = 0;
          PrologueA::template getActivation<ISA>(&aptr_cache, &acache_step, _param.paramA, m_remain, k_tail,
                                                 (blk_m + i + _config.loc[0]), iterk + k_paddedle, tmpcache,
                                                 _config.tmpcachesize);
          GemmCore::forward(aptr_cache, bptr_cache + k_paddedle * GemmCore::NTILE, cptr_cache, m_remain, n_padded,
                            GemmCore::KTILE, acache_step * sizeof(AType), bcache_stride, ccache_stride,
                            iterk + k_paddedle, tmpcache, _config.tmpcachesize);
        }
      }
    }
    Epilogue::template forward<ISA>(tmpC, _config.block[1], (_config.loc[0] + blk_m), _config.loc[1] + blk_n, blk_msize,
                                    blk_nsize, _param.paramC, tmpcache, _config.tmpcachesize);
  }
};

template <class _GemmCore_T, template <class _T> class _PrologueA_T, template <class _T> class _PrologueB_T,
          class _Epilogue_T>
class LauncherIntKBlock {
 public:
  using GemmCore = _GemmCore_T;
  static constexpr BTLA_ISA ISA = GemmCore::ISA;
  using PrologueA = _PrologueA_T<GemmCore>;
  using PrologueB = _PrologueB_T<GemmCore>;
  using Epilogue = _Epilogue_T;
  using AType = typename GemmCore::AType;
  using AParam = typename PrologueA::Param;
  using BType = typename GemmCore::BType;
  using BParam = typename PrologueB::Param;
  using CType = typename GemmCore::CType;
  using EpiParam = typename Epilogue::Param;
  using AccType = float;
  struct Param {
    const utils::GemmProblem problem;
    const AParam paramA;
    const BParam paramB;
    const EpiParam paramC;
  };

  class GEMVWrapper {
   public:
    static constexpr bool support() {
      if constexpr (!std::is_same_v<PrologueB, prologue_b::gemm::WeightKBlockNInteger<_GemmCore_T>>) {
        return false;
      }
      if constexpr (!std::is_same_v<PrologueA, prologue_a::gemm::ActivationF32KBlockQuantize<_GemmCore_T>> &&
                    !std::is_same_v<PrologueA, prologue_a::gemm::ShuffleActivationKBlockQuantizeF32<_GemmCore_T>>) {
        return false;
      }
      if constexpr (GemmCore::ISA == BTLA_ISA::AVX_VNNI) {
#if CompileAVXVNNI()
        if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_INT8_US_FP32) {
          static_assert(GemmCore::PACK_ROW == 4);
          return true;
        }
        if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_INT8_SS_FP32) {
          static_assert(GemmCore::PACK_ROW == 4);
          return true;
        }
#endif
      }
      if constexpr (GemmCore::ISA == BTLA_ISA::AVX2) {
#if CompileAVX2()
        if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_INT8_US_FP32) {
          static_assert(GemmCore::PACK_ROW == 4);
          return true;
        }
        if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_INT8_SS_FP32) {
          static_assert(GemmCore::PACK_ROW == 4);
          return true;
        }
#endif
      }
      if constexpr (GemmCore::ISA == BTLA_ISA::AVX512BW) {
#if CompileAVX512F()
        if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_INT8_US_FP32) {
          static_assert(GemmCore::PACK_ROW == 4);
          return true;
        }
#endif
      }
      if constexpr (GemmCore::ISA == BTLA_ISA::AVX512_VNNI || GemmCore::ISA == BTLA_ISA::AMX_INT8) {
#if CompileAVX512VNNI()
        if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_INT8_US_FP32) {
          static_assert(GemmCore::PACK_ROW == 4);
          return true;
        }
        if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_INT8_SS_FP32) {
          static_assert(GemmCore::PACK_ROW == 4);
          return true;
        }
#endif
      }
      return false;
    }
    static bool constexpr Reg32 = GemmCore::ISA >= BTLA_ISA::AVX512F;
    static int constexpr MaxGemvM = Reg32 ? 4 : 4;

    static bool implemented(const Param& _param) {
      bool impl = true;
      impl &= _param.paramB.packedW->mDType == BTLA_DTYPE::S4_CLIP ||
              _param.paramB.packedW->mDType == BTLA_DTYPE::S7_CLIP ||
              _param.paramB.packedW->mDType == BTLA_DTYPE::S6_CLIP ||
              _param.paramB.packedW->mDType == BTLA_DTYPE::S5_CLIP ||
              _param.paramB.packedW->mDType == BTLA_DTYPE::S3_CLIP ||
              _param.paramB.packedW->mDType == BTLA_DTYPE::S1_CLIP ||
              _param.paramB.packedW->mDType == BTLA_DTYPE::S2_CLIP;
      impl &= _param.paramB.packedW->mCorrection.mScaT == BTLA_DTYPE::F32 ||
              _param.paramB.packedW->mCorrection.mScaT == BTLA_DTYPE::F16 ||
              _param.paramB.packedW->mCorrection.mScaT == BTLA_DTYPE::BF16;
      impl &= _param.problem.dims[1] <= MaxGemvM;
      return impl;
    }

    template <typename ScaleT, int MTILE>
    static void gemv_kblock(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
      if constexpr (support()) {
        auto constexpr TmpSize = 16 * 1024LL;
        auto constexpr CSize = 8 * 1024LL;
        static_assert(CSize >= (MTILE * GemmCore::NTILE * sizeof(float)));
        auto StackTmp_ = alloca(TmpSize + CSize);
        auto StackTmp = utils::cpu_pointer_align<void>(StackTmp_);
        auto tmpc_ptr = reinterpret_cast<CType*>((char*)StackTmp + TmpSize);
        utils::GemvParamB<ScaleT> paramB = gemv_nbits::NBitsHelper::template createB<ScaleT>(_param.paramB.packedW);
        utils::GemvParamA paramA{
            _param.paramA.quan->template APtr<uint8_t>(), _param.paramA.quan->template SPtr<float>(),
            _param.paramA.quan->template ZPtr<uint8_t>(), _param.paramA.quan->mKPad, _param.paramA.quan->CStep()};

        int m = _param.problem.dims[1];
        int n = _param.problem.dims[2];
        int k = _param.problem.dims[3];
        int kblocksize = _param.problem.dims[4];
        gemv_nbits::NBitsHelper::template updateBNStep<ScaleT>(paramB, _config.loc[1]);
        int size_padded = utils::padto_le(_config.size[1], GemmCore::NTILE);
        int in = 0;
        for (; in < size_padded; in += GemmCore::NTILE) {
          if constexpr (std::is_same_v<AType, uint8_t>) {
            kernel::wrapper::GEMVWoqNBits::forward_u8s8_fp32<ISA, ScaleT, GemmCore::NTILE, MTILE>(
                paramA, paramB, tmpc_ptr, GemmCore::NTILE, k, kblocksize, StackTmp, TmpSize);
          } else if constexpr (std::is_same_v<AType, int8_t>) {
            kernel::wrapper::GEMVWoqNBits::forward_s8s8_fp32<ISA, ScaleT, GemmCore::NTILE, MTILE>(
                paramA, paramB, tmpc_ptr, GemmCore::NTILE, k, kblocksize, StackTmp, TmpSize);
          }
          Epilogue::template forward<ISA>(tmpc_ptr, GemmCore::NTILE, 0, _config.loc[1] + in, MTILE, GemmCore::NTILE,
                                          _param.paramC, StackTmp, TmpSize);
          gemv_nbits::NBitsHelper::template updateBNStep<ScaleT>(paramB, GemmCore::NTILE);
        }
        if (size_padded != _config.size[1]) {
          if constexpr (std::is_same_v<AType, uint8_t>) {
            kernel::wrapper::GEMVWoqNBits::forward_u8s8_fp32<ISA, ScaleT, GemmCore::NTILE, MTILE>(
                paramA, paramB, tmpc_ptr, GemmCore::NTILE, k, kblocksize, StackTmp, TmpSize);
          } else if constexpr (std::is_same_v<AType, int8_t>) {
            kernel::wrapper::GEMVWoqNBits::forward_s8s8_fp32<ISA, ScaleT, GemmCore::NTILE, MTILE>(
                paramA, paramB, tmpc_ptr, GemmCore::NTILE, k, kblocksize, StackTmp, TmpSize);
          }
          Epilogue::template forward<ISA>(tmpc_ptr, GemmCore::NTILE, 0, _config.loc[1] + in, MTILE,
                                          (_config.size[1] - in), _param.paramC, StackTmp, TmpSize);
        }
      }
    }

    static void gemv(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
      if constexpr (support()) {
        auto& m = _param.problem.dims[1];
        if (_param.paramB.packedW->SDtype() == BTLA_DTYPE::F32) {
          if (m == 1) gemv_kblock<float, 1>(_param, _config);
          if (m == 2) gemv_kblock<float, 2>(_param, _config);
          if (m == 3) gemv_kblock<float, 3>(_param, _config);
          if (m == 4) gemv_kblock<float, 4>(_param, _config);
          if constexpr (Reg32) {
            if (m == 5) gemv_kblock<float, 5>(_param, _config);
            if (m == 6) gemv_kblock<float, 6>(_param, _config);
            if (m == 7) gemv_kblock<float, 7>(_param, _config);
            if (m == 8) gemv_kblock<float, 8>(_param, _config);
          }
        } else if (_param.paramB.packedW->SDtype() == BTLA_DTYPE::BF16) {
          if (m == 1) gemv_kblock<utils::bf16, 1>(_param, _config);
          if (m == 2) gemv_kblock<utils::bf16, 2>(_param, _config);
          if (m == 3) gemv_kblock<utils::bf16, 3>(_param, _config);
          if (m == 4) gemv_kblock<utils::bf16, 4>(_param, _config);
          if constexpr (Reg32) {
            if (m == 5) gemv_kblock<utils::bf16, 5>(_param, _config);
            if (m == 6) gemv_kblock<utils::bf16, 6>(_param, _config);
            if (m == 7) gemv_kblock<utils::bf16, 7>(_param, _config);
            if (m == 8) gemv_kblock<utils::bf16, 8>(_param, _config);
          }
        } else if (_param.paramB.packedW->SDtype() == BTLA_DTYPE::F16) {
          if (m == 1) gemv_kblock<utils::fp16, 1>(_param, _config);
          if (m == 2) gemv_kblock<utils::fp16, 2>(_param, _config);
          if (m == 3) gemv_kblock<utils::fp16, 3>(_param, _config);
          if (m == 4) gemv_kblock<utils::fp16, 4>(_param, _config);
          if constexpr (Reg32) {
            if (m == 5) gemv_kblock<utils::fp16, 5>(_param, _config);
            if (m == 6) gemv_kblock<utils::fp16, 6>(_param, _config);
            if (m == 7) gemv_kblock<utils::fp16, 7>(_param, _config);
            if (m == 8) gemv_kblock<utils::fp16, 8>(_param, _config);
          }
        }
      }
    }
  };

  static void run(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
    if (GEMVWrapper::support() && GEMVWrapper::implemented(_param)) {
      GEMVWrapper::gemv(_param, _config);
    } else {
      gemm(_param, _config);
    }
    bestla::kernel::wrapper::ZeroReg::forward();
  }

 protected:
  static void gemm(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
    GemmCore::configure(_config.size[0], _config.size[1], _param.problem.dims[3]);
    auto StackTmp = alloca(_config.stacksize);
    auto tmpB = reinterpret_cast<BType*>(StackTmp);
    tmpB = utils::cpu_pointer_align(tmpB);
    auto tmpA = reinterpret_cast<AType*>(tmpB + static_cast<size_t>(_config.block[1]) * _config.block[2]);
    tmpA = utils::cpu_pointer_align(tmpA);
    auto tmpC = reinterpret_cast<AccType*>(tmpA + static_cast<size_t>(GemmCore::MTILE) * _config.block[2]);
    tmpC = utils::cpu_pointer_align(tmpC);
    auto tmpCache = reinterpret_cast<int8_t*>(tmpC + static_cast<size_t>(_config.block[0]) * _config.block[1]);
    tmpCache = utils::cpu_pointer_align(tmpCache);
    auto& KBlock = _param.problem.dims[4];
    assert(_config.block[2] >= KBlock);
    for (int itern = 0; itern < _config.size[1]; itern += _config.block[1]) {
      int n_remain = utils::remainsize(itern, _config.size[1], _config.block[1]);
      for (int iterm = 0; iterm < _config.size[0]; iterm += _config.block[0]) {
        int m_remain = utils::remainsize(iterm, _config.size[0], _config.block[0]);
        run_block(_param, _config, iterm, itern, m_remain, n_remain, tmpA, tmpB, tmpC, tmpCache);
      }
    }
  }

  // _config.block[2]%kblock==0
  // _config.block[2]>=kblock
  static void run_block(const Param& _param, const parallel::gemm::ThreadProblemBase& _config, int blk_m, int blk_n,
                        int blk_msize, int blk_nsize, AType* tmpA, BType* tmpB, AccType* tmpC, int8_t* tmpcache) {
    int n_padded = utils::padto(blk_nsize, GemmCore::NTILE);
    auto& K = _param.problem.dims[3];
    auto& KBlock = _param.problem.dims[4];
    assert(_config.block[2] % KBlock == 0);
    assert(_config.block[2] % GemmCore::KTILE == 0);
    // GemmCore: int8+int8=int32=>dequant to fp32
    // accumulate to tmpC
    // extra parameters: zpA, scaleA, scaleB, reduceB
    // zpA scaleA: [MTILE,kblk_perstep]
    // scaleB reduceB: [kblk_perstep, NStep]
    int kblk_perstep = utils::updiv(_config.block[2], KBlock);
    int tmp_ldsb = _config.block[1];
    int tmp_ldsa = kblk_perstep;
    auto zpA = reinterpret_cast<AType*>(tmpcache);
    zpA = utils::cpu_pointer_align(zpA);
    auto scaleA = reinterpret_cast<float*>(zpA + GemmCore::MTILE * tmp_ldsa);
    scaleA = utils::cpu_pointer_align(scaleA);
    auto scaleB = reinterpret_cast<float*>(scaleA + GemmCore::MTILE * tmp_ldsa);
    scaleB = utils::cpu_pointer_align(scaleB);
    auto reduceB = reinterpret_cast<int32_t*>(scaleB + _config.block[1] * tmp_ldsa);
    reduceB = utils::cpu_pointer_align(reduceB);
    auto tmp_ = reinterpret_cast<int8_t*>(reduceB + _config.block[1] * tmp_ldsa);
    tmp_ = utils::cpu_pointer_align(tmp_);

    for (int iterk = 0; iterk < K; iterk += _config.block[2]) {
      int k_remain = utils::remainsize(iterk, K, _config.block[2]);
      int k_padded = utils::padto(k_remain, GemmCore::KTILE);
      auto bptr_cache = tmpB;
      int bcache_step = 0;
      int ldsb_cache = tmp_ldsb;
      auto scaleB_cache = scaleB;
      auto reduceB_cache = reduceB;
      PrologueB::template getWeight<ISA>(&bptr_cache, &bcache_step, k_padded, n_padded, iterk, _config.loc[1] + blk_n,
                                         _param.paramB, tmp_, _config.tmpcachesize);
      PrologueB::template getScale<ISA>(&scaleB_cache, &ldsb_cache, k_padded, n_padded, iterk, _config.loc[1] + blk_n,
                                        _param.paramB, tmp_, _config.tmpcachesize);
      PrologueB::template getReduce<ISA>(&reduceB_cache, &ldsb_cache, k_padded, n_padded, iterk, _config.loc[1] + blk_n,
                                         _param.paramB, tmp_, _config.tmpcachesize);
      int bcache_stride = bcache_step * sizeof(BType);
      for (int i = 0; i < blk_msize; i += GemmCore::MTILE) {
        int m_remain = utils::remainsize(i, blk_msize, GemmCore::MTILE);
        auto cptr_cache = tmpC + i * _config.block[1];
        int ccache_stride = _config.block[1] * sizeof(CType);
        auto aptr_cache = tmpA;
        int acache_step = k_padded;
        auto zpA_cache = zpA;
        auto scaleA_cache = scaleA;
        int ldsa_cache = tmp_ldsa;
        PrologueA::template getActivation<ISA>(&aptr_cache, &acache_step, _param.paramA, m_remain, k_padded,
                                               (blk_m + i + _config.loc[0]), iterk, tmp_, _config.tmpcachesize);
        PrologueA::template getZp<ISA>(&zpA_cache, &ldsa_cache, _param.paramA, m_remain, k_padded,
                                       (blk_m + i + _config.loc[0]), iterk, tmp_, _config.tmpcachesize);
        PrologueA::template getScale<ISA>(&scaleA_cache, &ldsa_cache, _param.paramA, m_remain, k_padded,
                                          (blk_m + i + _config.loc[0]), iterk, tmp_, _config.tmpcachesize);
        GemmCore::forward(aptr_cache, bptr_cache, cptr_cache, zpA_cache, scaleA_cache, ldsa_cache, scaleB_cache,
                          reduceB_cache, ldsb_cache, m_remain, n_padded, k_padded, KBlock, acache_step * sizeof(AType),
                          bcache_stride, ccache_stride, iterk, tmp_, _config.tmpcachesize);
      }
    }
    Epilogue::template forward<ISA>(tmpC, _config.block[1], (_config.loc[0] + blk_m), _config.loc[1] + blk_n, blk_msize,
                                    blk_nsize, _param.paramC, tmpcache, _config.tmpcachesize);
  }
};

template <class _GemmCore_T, template <class _T> class _PrologueA_T, template <class _T> class _PrologueB_T,
          class _Epilogue_T>
class LauncherBaseV2 {
 public:
  using GemmCore = _GemmCore_T;
  static constexpr BTLA_ISA ISA = GemmCore::ISA;
  using PrologueA = _PrologueA_T<GemmCore>;
  using PrologueB = _PrologueB_T<GemmCore>;
  using Epilogue = _Epilogue_T;
  using AType = typename GemmCore::AType;
  using AParam = typename PrologueA::Param;
  using BType = typename GemmCore::BType;
  using BParam = typename PrologueB::Param;
  using CType = typename GemmCore::CType;
  using EpiParam = typename Epilogue::Param;
  struct Param {
    const utils::GemmProblem problem;
    const AParam paramA;
    const BParam paramB;
    const EpiParam paramC;
  };

  class GEMVWrapper {
   public:
    static constexpr bool support() {
      if constexpr (!std::is_same_v<PrologueB, prologue_b::gemm::WeightKBlockNInt<_GemmCore_T>>) {
        return false;
      }
      if constexpr (!std::is_same_v<PrologueA, prologue_a::gemm::ShuffleActivationKBlockBaseF32<_GemmCore_T>> &&
                    !std::is_same_v<PrologueA, prologue_a::gemm::ActivationKBlockBaseF32<_GemmCore_T>> &&
                    !std::is_same_v<PrologueA, prologue_a::gemm::ActivationF32KBlockQuantize<_GemmCore_T>> &&
                    !std::is_same_v<PrologueA, prologue_a::gemm::ShuffleActivationKBlockQuantizeF32<_GemmCore_T>> &&
                    !std::is_same_v<PrologueA, prologue_a::gemm::ActivationBase<_GemmCore_T>>) {
        return false;
      }

      if constexpr (GemmCore::ISA == BTLA_ISA::AVX2) {
#if CompileAVX2()
        if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_FP32) {
          static_assert(GemmCore::PACK_ROW == 1);
          return true;
        }
        if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_INT8_US_INT32) {
          static_assert(GemmCore::PACK_ROW == 4);
          return true;
        }
#endif
      }
      if constexpr (GemmCore::ISA == BTLA_ISA::AVX512_VNNI || GemmCore::ISA == BTLA_ISA::AMX_INT8) {
#if CompileAVX512VNNI()
        if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_INT8_US_INT32) {
          static_assert(GemmCore::PACK_ROW == 4);
          return true;
        }
#endif
      }
      if constexpr (GemmCore::ISA == BTLA_ISA::AVX_VNNI) {
#if CompileAVXVNNI()
        if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_INT8_US_INT32) {
          static_assert(GemmCore::PACK_ROW == 4);
          return true;
        }
#endif
      }
      if constexpr (GemmCore::ISA == BTLA_ISA::AVX512F) {
#if CompileAVX512F()
        if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_FP32) {
          static_assert(GemmCore::PACK_ROW == 1);
          return true;
        }
#endif
      }
      if constexpr (GemmCore::ISA == BTLA_ISA::AVX512BW) {
#if CompileAVX512F()
        if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_INT8_US_INT32) {
          static_assert(GemmCore::PACK_ROW == 4);
          return true;
        }
#endif
      }
      return false;
    }
    static bool constexpr Reg32 = GemmCore::ISA >= BTLA_ISA::AVX512F;
    static int constexpr MaxGemvM = Reg32 ? 4 : 4;
    static bool implemented(const Param& _param) {
      bool impl = true;
      impl &= _param.paramB.stor_->info_.dtype_ == BTLA_DTYPE::S4_CLIP ||
              _param.paramB.stor_->info_.dtype_ == BTLA_DTYPE::S3_CLIP ||
              _param.paramB.stor_->info_.dtype_ == BTLA_DTYPE::S7_CLIP ||
              _param.paramB.stor_->info_.dtype_ == BTLA_DTYPE::S6_CLIP ||
              _param.paramB.stor_->info_.dtype_ == BTLA_DTYPE::S5_CLIP ||
              _param.paramB.stor_->info_.dtype_ == BTLA_DTYPE::S1_CLIP ||
              _param.paramB.stor_->info_.dtype_ == BTLA_DTYPE::S2_CLIP;
      if constexpr (support()) {
        impl &= _param.paramB.stor_->corr_.sdtype_ == BTLA_DTYPE::F32 ||
                _param.paramB.stor_->corr_.sdtype_ == BTLA_DTYPE::F16 ||
                _param.paramB.stor_->corr_.sdtype_ == BTLA_DTYPE::BF16;
      }

      impl &= _param.problem.dims[1] <= MaxGemvM;
      return impl;
    }

    template <typename ScaleT, int MTILE>
    static void gemv_kblock(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
      if constexpr (support()) {
        auto constexpr TmpSize = 16 * 1024LL;
        auto constexpr CSize = 8 * 1024LL;
        auto StackTmp_ = alloca(TmpSize + CSize);
        auto StackTmp = utils::cpu_pointer_align<void>(StackTmp_);
        auto tmpc_ptr = reinterpret_cast<float*>((char*)StackTmp + TmpSize);
        static_assert(CSize >= (MTILE * GemmCore::NTILE * sizeof(float)));
        utils::GemvParamB<ScaleT> paramB = gemv_nbits::GemvWeight::template createB<ScaleT>(_param.paramB.stor_);
        int m = _param.problem.dims[1];
        int n = _param.problem.dims[2];
        int k = _param.problem.dims[3];
        int kblocksize = _param.problem.dims[4];
        gemv_nbits::GemvWeight::template updateBNStep<ScaleT>(paramB, _config.loc[1]);
        int size_padded = utils::padto_le(_config.size[1], GemmCore::NTILE);
        int in = 0;
        for (; in < size_padded; in += GemmCore::NTILE) {
          if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_INT8_US_INT32) {
            utils::GemvParamA paramA{
                _param.paramA.quan->template APtr<uint8_t>(), _param.paramA.quan->template SPtr<float>(),
                _param.paramA.quan->template ZPtr<uint8_t>(), _param.paramA.quan->mKPad, _param.paramA.quan->CStep()};
            kernel::wrapper::GEMVWoqNBits::forward_u8s8_fp32<ISA, ScaleT, GemmCore::NTILE, MTILE>(
                paramA, paramB, tmpc_ptr, GemmCore::NTILE, k, kblocksize, StackTmp, TmpSize);
            Epilogue::Fp32Epi::template forward<ISA>(tmpc_ptr, GemmCore::NTILE, 0, _config.loc[1] + in, MTILE,
                                                     GemmCore::NTILE, _param.paramC.param2, StackTmp, TmpSize);
          } else {
            const float* Aptr = _param.paramA.A;
            if constexpr (std::is_same_v<PrologueA, prologue_a::gemm::ShuffleActivationKBlockBaseF32<_GemmCore_T>>) {
              if (_param.paramA.reordered && _param.paramA.reordered->template APtr<float>()) {
                Aptr = _param.paramA.reordered->template APtr<float>();
              }
            }
            kernel::wrapper::GEMVWoqNBits::forward_fp32_fp32<ISA, ScaleT, GemmCore::NTILE, MTILE>(
                Aptr, _param.paramA.lda, paramB, tmpc_ptr, GemmCore::NTILE, k, kblocksize, StackTmp, TmpSize);
            Epilogue::template forward<ISA>(tmpc_ptr, GemmCore::NTILE, 0, _config.loc[1] + in, MTILE, GemmCore::NTILE,
                                            _param.paramC, StackTmp, TmpSize);
          }
          gemv_nbits::NBitsHelper::template updateBNStep<ScaleT>(paramB, GemmCore::NTILE);
        }
        if (size_padded != _config.size[1]) {
          if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_INT8_US_INT32) {
            utils::GemvParamA paramA{
                _param.paramA.quan->template APtr<uint8_t>(), _param.paramA.quan->template SPtr<float>(),
                _param.paramA.quan->template ZPtr<uint8_t>(), _param.paramA.quan->mKPad, _param.paramA.quan->CStep()};
            kernel::wrapper::GEMVWoqNBits::forward_u8s8_fp32<ISA, ScaleT, GemmCore::NTILE, MTILE>(
                paramA, paramB, tmpc_ptr, GemmCore::NTILE, k, kblocksize, StackTmp, TmpSize);
            Epilogue::Fp32Epi::template forward<ISA>(tmpc_ptr, GemmCore::NTILE, 0, _config.loc[1] + in, MTILE,
                                                     (_config.size[1] - in), _param.paramC.param2, StackTmp, TmpSize);
          } else {
            const float* Aptr = _param.paramA.A;
            if constexpr (std::is_same_v<PrologueA, prologue_a::gemm::ShuffleActivationKBlockBaseF32<_GemmCore_T>>) {
              if (_param.paramA.reordered && _param.paramA.reordered->template APtr<float>()) {
                Aptr = _param.paramA.reordered->template APtr<float>();
              }
            }
            kernel::wrapper::GEMVWoqNBits::forward_fp32_fp32<ISA, ScaleT, GemmCore::NTILE, MTILE>(
                Aptr, _param.paramA.lda, paramB, tmpc_ptr, GemmCore::NTILE, k, kblocksize, StackTmp, TmpSize);
            Epilogue::template forward<ISA>(tmpc_ptr, GemmCore::NTILE, 0, _config.loc[1] + in, MTILE,
                                            (_config.size[1] - in), _param.paramC, StackTmp, TmpSize);
          }
        }
      }
    }

    static void gemv(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
      if constexpr (support()) {
        assert(_param.problem.dims[4] > 0);
        auto& m = _param.problem.dims[1];
        if (_param.paramB.stor_->corr_.sdtype_ == BTLA_DTYPE::F32) {
          if (m == 1) gemv_kblock<float, 1>(_param, _config);
          if (m == 2) gemv_kblock<float, 2>(_param, _config);
          if (m == 3) gemv_kblock<float, 3>(_param, _config);
          if (m == 4) gemv_kblock<float, 4>(_param, _config);
          if constexpr (Reg32) {
            if (m == 5) gemv_kblock<float, 5>(_param, _config);
            if (m == 6) gemv_kblock<float, 6>(_param, _config);
            if (m == 7) gemv_kblock<float, 7>(_param, _config);
            if (m == 8) gemv_kblock<float, 8>(_param, _config);
          }
        } else if (_param.paramB.stor_->corr_.sdtype_ == BTLA_DTYPE::BF16) {
          if (m == 1) gemv_kblock<utils::bf16, 1>(_param, _config);
          if (m == 2) gemv_kblock<utils::bf16, 2>(_param, _config);
          if (m == 3) gemv_kblock<utils::bf16, 3>(_param, _config);
          if (m == 4) gemv_kblock<utils::bf16, 4>(_param, _config);
          if constexpr (Reg32) {
            if (m == 5) gemv_kblock<utils::bf16, 5>(_param, _config);
            if (m == 6) gemv_kblock<utils::bf16, 6>(_param, _config);
            if (m == 7) gemv_kblock<utils::bf16, 7>(_param, _config);
            if (m == 8) gemv_kblock<utils::bf16, 8>(_param, _config);
          }
        } else if (_param.paramB.stor_->corr_.sdtype_ == BTLA_DTYPE::F16) {
          if (m == 1) gemv_kblock<utils::fp16, 1>(_param, _config);
          if (m == 2) gemv_kblock<utils::fp16, 2>(_param, _config);
          if (m == 3) gemv_kblock<utils::fp16, 3>(_param, _config);
          if (m == 4) gemv_kblock<utils::fp16, 4>(_param, _config);
          if constexpr (Reg32) {
            if (m == 5) gemv_kblock<utils::fp16, 5>(_param, _config);
            if (m == 6) gemv_kblock<utils::fp16, 6>(_param, _config);
            if (m == 7) gemv_kblock<utils::fp16, 7>(_param, _config);
            if (m == 8) gemv_kblock<utils::fp16, 8>(_param, _config);
          }
        }
      }
    }
  };

  static void run(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
    if (GEMVWrapper::support() && GEMVWrapper::implemented(_param)) {
      GEMVWrapper::gemv(_param, _config);
    } else {
      gemm(_param, _config);
    }
    bestla::kernel::wrapper::ZeroReg::forward();
  }

 protected:
  static void gemm(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
    GemmCore::configure(_config.size[0], _config.size[1], _param.problem.dims[3]);
    auto StackTmp = alloca(_config.stacksize);
    auto tmpB = reinterpret_cast<BType*>(StackTmp);
    tmpB = utils::cpu_pointer_align(tmpB);
    auto tmpA = reinterpret_cast<AType*>(tmpB + static_cast<size_t>(_config.block[1]) * _config.block[2]);
    tmpA = utils::cpu_pointer_align(tmpA);
    auto tmpC = reinterpret_cast<CType*>(tmpA + static_cast<size_t>(GemmCore::MTILE) * _config.block[2]);
    tmpC = utils::cpu_pointer_align(tmpC);
    auto tmpCache = (void*)(tmpC + static_cast<size_t>(_config.block[0]) * _config.block[1]);
    tmpCache = utils::cpu_pointer_align(tmpCache);
    for (int itern = 0; itern < _config.size[1]; itern += _config.block[1]) {
      int n_remain = utils::remainsize(itern, _config.size[1], _config.block[1]);
      for (int iterm = 0; iterm < _config.size[0]; iterm += _config.block[0]) {
        int m_remain = utils::remainsize(iterm, _config.size[0], _config.block[0]);
        run_block(_param, _config, iterm, itern, m_remain, n_remain, tmpA, tmpB, tmpC, tmpCache);
      }
    }
  }

  static void run_block(const Param& _param, const parallel::gemm::ThreadProblemBase& _config, int blk_m, int blk_n,
                        int blk_msize, int blk_nsize, AType* tmpA, BType* tmpB, CType* tmpC, void* tmpcache) {
    int n_padded = utils::padto(blk_nsize, GemmCore::NTILE);
    auto& K = _param.problem.dims[3];
    for (int iterk = 0; iterk < _param.problem.dims[3]; iterk += _config.block[2]) {
      int k_remain = utils::remainsize(iterk, K, _config.block[2]);
      int k_padded = utils::padto(k_remain, GemmCore::KTILE);
      int k_paddedle = utils::padto_le(k_remain, GemmCore::KTILE);
      auto bptr_cache = tmpB;
      int bcache_step = 0;
      PrologueB::template get_weight<ISA>(&bptr_cache, &bcache_step, k_padded, n_padded, iterk, _config.loc[1] + blk_n,
                                          _param.paramB, tmpcache, _config.tmpcachesize);
      int bcache_stride = bcache_step * sizeof(BType);
      for (int i = 0; i < blk_msize; i += GemmCore::MTILE) {
        int m_remain = utils::remainsize(i, blk_msize, GemmCore::MTILE);
        auto cptr_cache = tmpC + i * _config.block[1];
        int ccache_stride = _config.block[1] * sizeof(CType);
        if (k_paddedle) {
          AType* aptr_cache = tmpA;
          int acache_step = 0;
          PrologueA::template getActivation<ISA>(&aptr_cache, &acache_step, _param.paramA, m_remain, k_paddedle,
                                                 (blk_m + i + _config.loc[0]), iterk, tmpcache, _config.tmpcachesize);
          GemmCore::forward(aptr_cache, bptr_cache, cptr_cache, m_remain, n_padded, k_paddedle,
                            acache_step * sizeof(AType), bcache_stride, ccache_stride, iterk, tmpcache,
                            _config.tmpcachesize);
        }
        int k_tail = k_remain - k_paddedle;
        if (k_tail) {
          AType* aptr_cache = tmpA;
          int acache_step = 0;
          PrologueA::template getActivation<ISA>(&aptr_cache, &acache_step, _param.paramA, m_remain, k_tail,
                                                 (blk_m + i + _config.loc[0]), iterk + k_paddedle, tmpcache,
                                                 _config.tmpcachesize);
          GemmCore::forward(aptr_cache, bptr_cache + k_paddedle * GemmCore::NTILE, cptr_cache, m_remain, n_padded,
                            GemmCore::KTILE, acache_step * sizeof(AType), bcache_stride, ccache_stride,
                            iterk + k_paddedle, tmpcache, _config.tmpcachesize);
        }
      }
    }
    Epilogue::template forward<ISA>(tmpC, _config.block[1], (_config.loc[0] + blk_m), _config.loc[1] + blk_n, blk_msize,
                                    blk_nsize, _param.paramC, tmpcache, _config.tmpcachesize);
  }
};

template <class _GemmCore_T, template <class _T> class _PrologueA_T, template <class _T> class _PrologueB_T,
          class _Epilogue_T>
class LauncherIntKBlockV2 {
 public:
  using GemmCore = _GemmCore_T;
  static constexpr BTLA_ISA ISA = GemmCore::ISA;
  using PrologueA = _PrologueA_T<GemmCore>;
  using PrologueB = _PrologueB_T<GemmCore>;
  using Epilogue = _Epilogue_T;
  using AType = typename GemmCore::AType;
  using AParam = typename PrologueA::Param;
  using BType = typename GemmCore::BType;
  using BParam = typename PrologueB::Param;
  using CType = typename GemmCore::CType;
  using EpiParam = typename Epilogue::Param;
  using AccType = float;
  struct Param {
    const utils::GemmProblem problem;
    const AParam paramA;
    const BParam paramB;
    const EpiParam paramC;
  };

  class GEMVWrapper {
   public:
    static constexpr bool support() {
      if constexpr (!std::is_same_v<PrologueB, prologue_b::gemm::WeightKBlockNInt<_GemmCore_T>>) {
        return false;
      }
      if constexpr (!std::is_same_v<PrologueA, prologue_a::gemm::ActivationF32KBlockQuantize<_GemmCore_T>> &&
                    !std::is_same_v<PrologueA, prologue_a::gemm::ShuffleActivationKBlockQuantizeF32<_GemmCore_T>>) {
        return false;
      }
      if constexpr (GemmCore::ISA == BTLA_ISA::AVX_VNNI) {
#if CompileAVXVNNI()
        if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_INT8_US_FP32) {
          static_assert(GemmCore::PACK_ROW == 4);
          return true;
        }
        if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_INT8_SS_FP32) {
          static_assert(GemmCore::PACK_ROW == 4);
          return true;
        }
#endif
      }
      if constexpr (GemmCore::ISA == BTLA_ISA::AVX2) {
#if CompileAVX2()
        if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_INT8_US_FP32) {
          static_assert(GemmCore::PACK_ROW == 4);
          return true;
        }
        if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_INT8_SS_FP32) {
          static_assert(GemmCore::PACK_ROW == 4);
          return true;
        }
#endif
      }
      if constexpr (GemmCore::ISA == BTLA_ISA::AVX512BW) {
#if CompileAVX512F()
        if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_INT8_US_FP32) {
          static_assert(GemmCore::PACK_ROW == 4);
          return true;
        }
#endif
      }
      if constexpr (GemmCore::ISA == BTLA_ISA::AVX512_VNNI || GemmCore::ISA == BTLA_ISA::AMX_INT8) {
#if CompileAVX512VNNI()
        if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_INT8_US_FP32) {
          static_assert(GemmCore::PACK_ROW == 4);
          return true;
        }
        if constexpr (GemmCore::COMP == bestla::gemm::CompType::COMP_INT8_SS_FP32) {
          static_assert(GemmCore::PACK_ROW == 4);
          return true;
        }
#endif
      }
      return false;
    }
    static bool constexpr Reg32 = GemmCore::ISA >= BTLA_ISA::AVX512F;
    static int constexpr MaxGemvM = Reg32 ? 4 : 4;

    static bool implemented(const Param& _param) {
      bool impl = true;
      impl &= _param.paramB.stor_->info_.dtype_ == BTLA_DTYPE::S4_CLIP ||
              _param.paramB.stor_->info_.dtype_ == BTLA_DTYPE::S3_CLIP ||
              _param.paramB.stor_->info_.dtype_ == BTLA_DTYPE::S7_CLIP ||
              _param.paramB.stor_->info_.dtype_ == BTLA_DTYPE::S6_CLIP ||
              _param.paramB.stor_->info_.dtype_ == BTLA_DTYPE::S5_CLIP ||
              _param.paramB.stor_->info_.dtype_ == BTLA_DTYPE::S1_CLIP ||
              _param.paramB.stor_->info_.dtype_ == BTLA_DTYPE::S2_CLIP;
      if constexpr (support()) {
        impl &= _param.paramB.stor_->corr_.sdtype_ == BTLA_DTYPE::F32 ||
                _param.paramB.stor_->corr_.sdtype_ == BTLA_DTYPE::F16 ||
                _param.paramB.stor_->corr_.sdtype_ == BTLA_DTYPE::BF16;
      }
      impl &= _param.problem.dims[1] <= MaxGemvM;
      return impl;
    }

    template <typename ScaleT, int MTILE>
    static void gemv_kblock(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
      if constexpr (support()) {
        auto constexpr TmpSize = 16 * 1024LL;
        auto constexpr CSize = 8 * 1024LL;
        static_assert(CSize >= (MTILE * GemmCore::NTILE * sizeof(float)));
        auto StackTmp_ = alloca(TmpSize + CSize);
        auto StackTmp = utils::cpu_pointer_align<void>(StackTmp_);
        auto tmpc_ptr = reinterpret_cast<CType*>((char*)StackTmp + TmpSize);
        utils::GemvParamB<ScaleT> paramB = gemv_nbits::GemvWeight::template createB<ScaleT>(_param.paramB.stor_);
        utils::GemvParamA paramA{
            _param.paramA.quan->template APtr<uint8_t>(), _param.paramA.quan->template SPtr<float>(),
            _param.paramA.quan->template ZPtr<uint8_t>(), _param.paramA.quan->mKPad, _param.paramA.quan->CStep()};

        int m = _param.problem.dims[1];
        int n = _param.problem.dims[2];
        int k = _param.problem.dims[3];
        int kblocksize = _param.problem.dims[4];
        gemv_nbits::GemvWeight::template updateBNStep<ScaleT>(paramB, _config.loc[1]);
        int size_padded = utils::padto_le(_config.size[1], GemmCore::NTILE);
        int in = 0;
        for (; in < size_padded; in += GemmCore::NTILE) {
          if constexpr (std::is_same_v<AType, uint8_t>) {
            kernel::wrapper::GEMVWoqNBits::forward_u8s8_fp32<ISA, ScaleT, GemmCore::NTILE, MTILE>(
                paramA, paramB, tmpc_ptr, GemmCore::NTILE, k, kblocksize, StackTmp, TmpSize);
          } else if constexpr (std::is_same_v<AType, int8_t>) {
            kernel::wrapper::GEMVWoqNBits::forward_s8s8_fp32<ISA, ScaleT, GemmCore::NTILE, MTILE>(
                paramA, paramB, tmpc_ptr, GemmCore::NTILE, k, kblocksize, StackTmp, TmpSize);
          }
          Epilogue::template forward<ISA>(tmpc_ptr, GemmCore::NTILE, 0, _config.loc[1] + in, MTILE, GemmCore::NTILE,
                                          _param.paramC, StackTmp, TmpSize);
          gemv_nbits::NBitsHelper::template updateBNStep<ScaleT>(paramB, GemmCore::NTILE);
        }
        if (size_padded != _config.size[1]) {
          if constexpr (std::is_same_v<AType, uint8_t>) {
            kernel::wrapper::GEMVWoqNBits::forward_u8s8_fp32<ISA, ScaleT, GemmCore::NTILE, MTILE>(
                paramA, paramB, tmpc_ptr, GemmCore::NTILE, k, kblocksize, StackTmp, TmpSize);
          } else if constexpr (std::is_same_v<AType, int8_t>) {
            kernel::wrapper::GEMVWoqNBits::forward_s8s8_fp32<ISA, ScaleT, GemmCore::NTILE, MTILE>(
                paramA, paramB, tmpc_ptr, GemmCore::NTILE, k, kblocksize, StackTmp, TmpSize);
          }
          Epilogue::template forward<ISA>(tmpc_ptr, GemmCore::NTILE, 0, _config.loc[1] + in, MTILE,
                                          (_config.size[1] - in), _param.paramC, StackTmp, TmpSize);
        }
      }
    }

    static void gemv(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
      if constexpr (support()) {
        auto& m = _param.problem.dims[1];
        if (_param.paramB.stor_->corr_.sdtype_ == BTLA_DTYPE::F32) {
          if (m == 1) gemv_kblock<float, 1>(_param, _config);
          if (m == 2) gemv_kblock<float, 2>(_param, _config);
          if (m == 3) gemv_kblock<float, 3>(_param, _config);
          if (m == 4) gemv_kblock<float, 4>(_param, _config);
          if constexpr (Reg32) {
            if (m == 5) gemv_kblock<float, 5>(_param, _config);
            if (m == 6) gemv_kblock<float, 6>(_param, _config);
            if (m == 7) gemv_kblock<float, 7>(_param, _config);
            if (m == 8) gemv_kblock<float, 8>(_param, _config);
          }
        } else if (_param.paramB.stor_->corr_.sdtype_ == BTLA_DTYPE::BF16) {
          if (m == 1) gemv_kblock<utils::bf16, 1>(_param, _config);
          if (m == 2) gemv_kblock<utils::bf16, 2>(_param, _config);
          if (m == 3) gemv_kblock<utils::bf16, 3>(_param, _config);
          if (m == 4) gemv_kblock<utils::bf16, 4>(_param, _config);
          if constexpr (Reg32) {
            if (m == 5) gemv_kblock<utils::bf16, 5>(_param, _config);
            if (m == 6) gemv_kblock<utils::bf16, 6>(_param, _config);
            if (m == 7) gemv_kblock<utils::bf16, 7>(_param, _config);
            if (m == 8) gemv_kblock<utils::bf16, 8>(_param, _config);
          }
        } else if (_param.paramB.stor_->corr_.sdtype_ == BTLA_DTYPE::F16) {
          if (m == 1) gemv_kblock<utils::fp16, 1>(_param, _config);
          if (m == 2) gemv_kblock<utils::fp16, 2>(_param, _config);
          if (m == 3) gemv_kblock<utils::fp16, 3>(_param, _config);
          if (m == 4) gemv_kblock<utils::fp16, 4>(_param, _config);
          if constexpr (Reg32) {
            if (m == 5) gemv_kblock<utils::fp16, 5>(_param, _config);
            if (m == 6) gemv_kblock<utils::fp16, 6>(_param, _config);
            if (m == 7) gemv_kblock<utils::fp16, 7>(_param, _config);
            if (m == 8) gemv_kblock<utils::fp16, 8>(_param, _config);
          }
        }
      }
    }
  };

  static void run(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
    if (GEMVWrapper::support() && GEMVWrapper::implemented(_param)) {
      GEMVWrapper::gemv(_param, _config);
    } else {
      gemm(_param, _config);
    }
    bestla::kernel::wrapper::ZeroReg::forward();
  }

 protected:
  static void gemm(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
    GemmCore::configure(_config.size[0], _config.size[1], _param.problem.dims[3]);
    auto StackTmp = alloca(_config.stacksize);
    auto tmpB = reinterpret_cast<BType*>(StackTmp);
    tmpB = utils::cpu_pointer_align(tmpB);
    auto tmpA = reinterpret_cast<AType*>(tmpB + static_cast<size_t>(_config.block[1]) * _config.block[2]);
    tmpA = utils::cpu_pointer_align(tmpA);
    auto tmpC = reinterpret_cast<AccType*>(tmpA + static_cast<size_t>(GemmCore::MTILE) * _config.block[2]);
    tmpC = utils::cpu_pointer_align(tmpC);
    auto tmpCache = reinterpret_cast<int8_t*>(tmpC + static_cast<size_t>(_config.block[0]) * _config.block[1]);
    tmpCache = utils::cpu_pointer_align(tmpCache);
    auto& KBlock = _param.problem.dims[4];
    assert(_config.block[2] >= KBlock);
    for (int itern = 0; itern < _config.size[1]; itern += _config.block[1]) {
      int n_remain = utils::remainsize(itern, _config.size[1], _config.block[1]);
      for (int iterm = 0; iterm < _config.size[0]; iterm += _config.block[0]) {
        int m_remain = utils::remainsize(iterm, _config.size[0], _config.block[0]);
        run_block(_param, _config, iterm, itern, m_remain, n_remain, tmpA, tmpB, tmpC, tmpCache);
      }
    }
  }

  // _config.block[2]%kblock==0
  // _config.block[2]>=kblock
  static void run_block(const Param& _param, const parallel::gemm::ThreadProblemBase& _config, int blk_m, int blk_n,
                        int blk_msize, int blk_nsize, AType* tmpA, BType* tmpB, AccType* tmpC, int8_t* tmpcache) {
    int n_padded = utils::padto(blk_nsize, GemmCore::NTILE);
    auto& K = _param.problem.dims[3];
    auto& KBlock = _param.problem.dims[4];
    assert(_config.block[2] % KBlock == 0);
    assert(_config.block[2] % GemmCore::KTILE == 0);
    // GemmCore: int8+int8=int32=>dequant to fp32
    // accumulate to tmpC
    // extra parameters: zpA, scaleA, scaleB, reduceB
    // zpA scaleA: [MTILE,kblk_perstep]
    // scaleB reduceB: [kblk_perstep, NStep]
    int kblk_perstep = utils::updiv(_config.block[2], KBlock);
    int tmp_ldsb = _config.block[1];
    int tmp_ldsa = kblk_perstep;
    auto zpA = reinterpret_cast<AType*>(tmpcache);
    zpA = utils::cpu_pointer_align(zpA);
    auto scaleA = reinterpret_cast<float*>(zpA + GemmCore::MTILE * tmp_ldsa);
    scaleA = utils::cpu_pointer_align(scaleA);
    auto scaleB = reinterpret_cast<float*>(scaleA + GemmCore::MTILE * tmp_ldsa);
    scaleB = utils::cpu_pointer_align(scaleB);
    auto reduceB = reinterpret_cast<int32_t*>(scaleB + _config.block[1] * tmp_ldsa);
    reduceB = utils::cpu_pointer_align(reduceB);
    auto tmp_ = reinterpret_cast<int8_t*>(reduceB + _config.block[1] * tmp_ldsa);
    tmp_ = utils::cpu_pointer_align(tmp_);

    for (int iterk = 0; iterk < K; iterk += _config.block[2]) {
      int k_remain = utils::remainsize(iterk, K, _config.block[2]);
      int k_padded = utils::padto(k_remain, GemmCore::KTILE);
      auto bptr_cache = tmpB;
      int bcache_step = 0;
      int ldsb_cache = tmp_ldsb;
      auto scaleB_cache = scaleB;
      auto reduceB_cache = reduceB;
      PrologueB::template get_weight<ISA>(&bptr_cache, &bcache_step, k_padded, n_padded, iterk, _config.loc[1] + blk_n,
                                          _param.paramB, tmp_, _config.tmpcachesize);
      PrologueB::template get_scale<ISA>(&scaleB_cache, &ldsb_cache, k_padded, n_padded, iterk, _config.loc[1] + blk_n,
                                         _param.paramB, tmp_, _config.tmpcachesize);
      PrologueB::template get_reduce<ISA>(&reduceB_cache, &ldsb_cache, k_padded, n_padded, iterk,
                                          _config.loc[1] + blk_n, _param.paramB, tmp_, _config.tmpcachesize);
      int bcache_stride = bcache_step * sizeof(BType);
      for (int i = 0; i < blk_msize; i += GemmCore::MTILE) {
        int m_remain = utils::remainsize(i, blk_msize, GemmCore::MTILE);
        auto cptr_cache = tmpC + i * _config.block[1];
        int ccache_stride = _config.block[1] * sizeof(CType);
        auto aptr_cache = tmpA;
        int acache_step = k_padded;
        auto zpA_cache = zpA;
        auto scaleA_cache = scaleA;
        int ldsa_cache = tmp_ldsa;
        PrologueA::template getActivation<ISA>(&aptr_cache, &acache_step, _param.paramA, m_remain, k_padded,
                                               (blk_m + i + _config.loc[0]), iterk, tmp_, _config.tmpcachesize);
        PrologueA::template getZp<ISA>(&zpA_cache, &ldsa_cache, _param.paramA, m_remain, k_padded,
                                       (blk_m + i + _config.loc[0]), iterk, tmp_, _config.tmpcachesize);
        PrologueA::template getScale<ISA>(&scaleA_cache, &ldsa_cache, _param.paramA, m_remain, k_padded,
                                          (blk_m + i + _config.loc[0]), iterk, tmp_, _config.tmpcachesize);
        GemmCore::forward(aptr_cache, bptr_cache, cptr_cache, zpA_cache, scaleA_cache, ldsa_cache, scaleB_cache,
                          reduceB_cache, ldsb_cache, m_remain, n_padded, k_padded, KBlock, acache_step * sizeof(AType),
                          bcache_stride, ccache_stride, iterk, tmp_, _config.tmpcachesize);
      }
    }
    Epilogue::template forward<ISA>(tmpC, _config.block[1], (_config.loc[0] + blk_m), _config.loc[1] + blk_n, blk_msize,
                                    blk_nsize, _param.paramC, tmpcache, _config.tmpcachesize);
  }
};

}  // namespace gemm
}  // namespace wrapper
}  // namespace bestla
