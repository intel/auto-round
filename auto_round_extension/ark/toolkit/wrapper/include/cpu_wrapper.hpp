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
#include "bestla/bestla.h"
#include "bestla/bestla_prologue_a.h"
#include "bestla/bestla_epilogue.h"
#include "bestla/bestla_gemm.h"
#include "bestla/bestla_gemm_kblock.h"
#include "bestla/bestla_device.h"
#include "bestla/bestla_wrapper.h"

namespace bestla {
template <class GemmCore_T, template <class> class Wei_T>
using tLauncher_Fp_F32F32 = wrapper::gemm::LauncherBase<GemmCore_T, prologue_a::gemm::ShuffleActivationKBlockBaseF32,
                                                        Wei_T, epilogue::gemm::AccumulatorWriteBackFp32>;

template <class GemmCore_T, template <class> class Wei_T>
using tLauncher_Int8_F32F32 =
    wrapper::gemm::LauncherIntKBlock<GemmCore_T, prologue_a::gemm::ShuffleActivationKBlockQuantizeF32, Wei_T,
                                     epilogue::gemm::AccumulatorWriteBackFp32>;

using PcWriteBackF32 = epilogue::gemm::PcKBlockCompInt8Epilogue<epilogue::gemm::AccumulatorWriteBackFp32>;

template <class GemmCore_T, template <class> class Wei_T>
using tLauncher_Int8Pc_F32F32 =
    wrapper::gemm::LauncherBase<GemmCore_T, prologue_a::gemm::ShuffleActivationKBlockQuantizeF32, Wei_T,
                                PcWriteBackF32>;

using tAVX2 = gemm::SCoreRowNAvx2<24, 4>;
using tAVX2_VNNI = gemm::ICoreRowNAvx2vnni<24, 4>;
using tAVX_VNNI = gemm::ICoreRowNAvxvnni<24, 4>;
using tAVX512F = gemm::SCoreRowNAvx512f<48, 8>;
using tAVX512BW = gemm::ICoreRowNAvx512bw<48, 8>;
using tAVX512_VNNI = gemm::ICoreRowNAvx512vnni<48, 8>;
using tAMX_BF16 = gemm::HCoreRowNAmxbf16<64, 16>;
using tAMX_FP16 = gemm::HCoreRowNAmxfp16<64, 16>;
using tAVX512_BF16 = gemm::HCoreRowNAvx512bf16<48, 8>;
using tAVX512_FP16 = gemm::HCoreRowNAvx512fp16<96, 8>;
using tAMX_INT8_US = gemm::ICoreRowNAmxint8<64, 16>;
using tAMX_INT8_SS = gemm::ICoreRowNAmxint8SS<64, 16>;

using tAVX2_VNNI_KBlock = gemm::KblockGemmAvx2vnni<24, 2>;
using tAVX_VNNI_KBlock = gemm::KblockGemmAvxvnni<24, 2>;
using tAVX512BW_KBlock = gemm::KblockGemmAvx512bw<48, 4>;
using tAVX512_VNNI_KBlock = gemm::KblockGemmAvx512vnni<48, 4>;
using tAMX_INT8_US_KBlock = gemm::KblockGemmAmxint8<64, 16>;
using tAMX_INT8_SS_KBlock = gemm::KblockGemmAmxint8SS<64, 16>;

template <class GC_T>
using tWeiNInt = prologue_b::gemm::WeightKBlockNInteger<GC_T>;
template <class GC_T>
using tWeiNFloat = prologue_b::gemm::WeightKBlockNFloat<GC_T>;

template <class GC_T>
using tActKBaseF32 = prologue_a::gemm::ShuffleActivationKBlockBaseF32<GC_T>;

constexpr uint64_t Fp32Cores[] = {tAVX2::ID, tAVX512F::ID};
constexpr uint64_t Bf16Cores[] = {tAMX_BF16::ID, tAVX512_BF16::ID};
constexpr uint64_t Fp16Cores[] = {tAVX512_FP16::ID, tAMX_FP16::ID};
constexpr uint64_t Int8Cores[] = {tAVX2::ID,        tAVX_VNNI::ID,    tAVX2_VNNI_KBlock::ID, tAVX512BW_KBlock::ID,
                                  tAVX512_VNNI::ID, tAMX_INT8_US::ID, tAMX_INT8_SS::ID};
constexpr uint64_t FloatCores[] = {tAVX2::ID, tAVX512F::ID, tAMX_BF16::ID, tAVX512_FP16::ID, tAMX_FP16::ID};
constexpr uint64_t AllKBlockCores[] = {tAVX2::ID,
                                       tAVX512F::ID,
                                       tAMX_BF16::ID,
                                       tAMX_FP16::ID,
                                       tAVX512_FP16::ID,
                                       tAVX2_VNNI_KBlock::ID,
                                       tAVX_VNNI_KBlock::ID,
                                       tAVX512BW_KBlock::ID,
                                       tAVX512_VNNI_KBlock::ID,
                                       tAMX_INT8_US_KBlock::ID,
                                       tAMX_INT8_SS_KBlock::ID};

static inline bool can_comps8(ark::QuantParam* p) {
  if (p->asym && p->weight_type == BTLA_DTYPE::S8) return false;
  if (p->blocksize == -1 || p->blocksize == p->k) return true;
  if (p->blocksize % 4 == 0) return true;
  return false;
}

static inline uint64_t BTLAGemmCoreId(ark::QuantParam* p) {
  GetCPUDevice();
  using namespace bestla;
  if (!_cd->INTEL()) {
    return tAVX2::ID;
  }
  switch (p->compute_type) {
    case BTLA_DTYPE::EleBitsUndef:  // from low precision to high precision
      [[fallthrough]];
    case BTLA_DTYPE::S8:
      if (can_comps8(p)) {
        if (_cd->AMX_INT8()) {
          if (p->blocksize == -1) {
            return tAMX_INT8_SS::ID;
          } else if (p->blocksize % tAMX_INT8_SS_KBlock::KTILE == 0) {
            return tAMX_INT8_SS_KBlock::ID;
          }
        }
        if (_cd->AVX512_VNNI()) {
          if (p->blocksize == -1) {
            return tAVX512_VNNI::ID;
          } else if (p->blocksize % tAVX512_VNNI_KBlock::KTILE == 0) {
            return tAVX512_VNNI_KBlock::ID;
          }
        }
        if (_cd->AVX_VNNI()) {
          if (p->blocksize == -1) {
            return tAVX_VNNI::ID;
          } else if (p->blocksize % tAVX_VNNI_KBlock::KTILE == 0) {
            return tAVX_VNNI_KBlock::ID;
          }
        }
      }
      [[fallthrough]];
    case BTLA_DTYPE::F16:
      if (_cd->AMX_FP16() && (p->blocksize % tAMX_FP16::KTILE == 0)) {
        return tAMX_FP16::ID;
      }
      [[fallthrough]];
    case BTLA_DTYPE::BF16:
      if (_cd->AMX_BF16() && (p->blocksize % tAMX_BF16::KTILE == 0)) {
        return tAMX_BF16::ID;
      }
      [[fallthrough]];
    case BTLA_DTYPE::F32:
      if (_cd->AVX512F()) {
        return tAVX512F::ID;
      }
      if (_cd->AVX2()) {
        return tAVX2::ID;
      }
      [[fallthrough]];
    default:
      return 0;
  }
}

static inline uint64_t BTLAGemmCoreId(BTLA_DTYPE dt) {
  GetCPUDevice();
  using namespace bestla;
  if (!_cd->INTEL()) {
    return tAVX2::ID;
  }
  switch (dt) {
    case BTLA_DTYPE::EleBitsUndef:  // from low precision to high precision
      [[fallthrough]];
    case BTLA_DTYPE::S8:
      if (_cd->AMX_INT8()) {
        return tAMX_INT8_SS::ID;
      }
      if (_cd->AVX512_VNNI()) {
        return tAVX512_VNNI::ID;
      }
      if (_cd->AVX_VNNI()) {
        return tAVX_VNNI::ID;
      }
      [[fallthrough]];
    case BTLA_DTYPE::F16:
      if (_cd->AMX_FP16()) {
        return tAMX_FP16::ID;
      }
      [[fallthrough]];
    case BTLA_DTYPE::BF16:
      if (_cd->AMX_BF16()) {
        return tAMX_BF16::ID;
      }
      [[fallthrough]];
    case BTLA_DTYPE::F32:
      if (_cd->AVX512F()) {
        return tAVX512F::ID;
      }
      if (_cd->AVX2()) {
        return tAVX2::ID;
      }
      [[fallthrough]];
    default:
      return 0;
  }
}

template <class GC_T>
static inline size_t BTLAGemmPackBSizeT(const ark::QuantParam* p) {
  using ProB = bestla::prologue_b::gemm::WeightKBlockNInt<GC_T>;
  auto stor = ProB::create_storage(p->n, p->k, p->blocksize, p->weight_type, p->scale_type, p->asym);
  return stor.total_size();
}

static inline size_t BTLAGemmPackBSize(uint64_t coreID, const ark::QuantParam* p) {
  using namespace bestla;
  switch (coreID) {
    case tAMX_INT8_SS::ID:
      return BTLAGemmPackBSizeT<tAMX_INT8_SS>(p);
    case tAMX_INT8_SS_KBlock::ID:
      return BTLAGemmPackBSizeT<tAMX_INT8_SS_KBlock>(p);
    case tAVX512_VNNI::ID:
      return BTLAGemmPackBSizeT<tAVX512_VNNI>(p);
    case tAVX512_VNNI_KBlock::ID:
      return BTLAGemmPackBSizeT<tAVX512_VNNI_KBlock>(p);
    case tAVX_VNNI::ID:
      return BTLAGemmPackBSizeT<tAVX_VNNI>(p);
    case tAVX_VNNI_KBlock::ID:
      return BTLAGemmPackBSizeT<tAVX_VNNI_KBlock>(p);
    case tAMX_FP16::ID:
      return BTLAGemmPackBSizeT<tAMX_FP16>(p);
    case tAMX_BF16::ID:
      return BTLAGemmPackBSizeT<tAMX_BF16>(p);
    case tAVX512F::ID:
      return BTLAGemmPackBSizeT<tAVX512F>(p);
    case tAVX2::ID:
      return BTLAGemmPackBSizeT<tAVX2>(p);
    default:
      return 0;
  }
  return 0;
}

static inline BTLA_DTYPE BTLAGemmBaseDType(size_t N, size_t K, uint64_t coreID) {
  using namespace bestla;
  switch (coreID) {
    case tAMX_FP16::ID:
      return BTLA_DTYPE::F16;
    case tAMX_BF16::ID:
      return BTLA_DTYPE::BF16;
    case tAVX512F::ID:
    case tAVX2::ID:
      return BTLA_DTYPE::F32;
    default:
      break;
  }
  return BTLA_DTYPE::EleBitsUndef;
}

template <class GC_T>
static inline void BTLAGemmPackQBT(const int8_t* QB, const float* scales, const int8_t* zero_points, void* PackedBuf,
                                   const ark::QuantParam* p, void* ThreadPool) {
  using ProB = bestla::prologue_b::gemm::WeightKBlockNInt<GC_T>;
  auto stor = ProB::create_storage(p->n, p->k, p->blocksize, p->weight_type, p->scale_type, p->asym);
  auto wptr = (int8_t*)PackedBuf;
  auto data_ptr = stor.serialize(wptr);
  stor.set_buffers(data_ptr);
  auto thd = (bestla::parallel::IThreading*)ThreadPool;
  ProB::pack_qweight(p->n, p->k, QB, p->n, scales, zero_points, stor, thd);
}

template <class GC_T>
static inline void BTLAGemmUnpackQBT(storage::gemm::StorageWeightNInt* B, float* DQB, void* ThreadPool) {
  using ProB = bestla::prologue_b::gemm::WeightKBlockNInt<GC_T>;
  auto thd = (bestla::parallel::IThreading*)ThreadPool;

  ProB::unpack_weight(B->info_.n_, B->info_.k_, *B, DQB, B->info_.n_, thd);
}

static inline void BTLAGemmPackQB(void* PackedBuf, const int8_t* QB, const float* scales, const int8_t* zero_points,
                                  uint64_t coreID, const ark::QuantParam* p, void* ThreadPool) {
  using namespace bestla;
  switch (coreID) {
    case tAMX_INT8_SS::ID:
      return BTLAGemmPackQBT<tAMX_INT8_SS>(QB, scales, zero_points, PackedBuf, p, ThreadPool);
    case tAMX_INT8_SS_KBlock::ID:
      return BTLAGemmPackQBT<tAMX_INT8_SS_KBlock>(QB, scales, zero_points, PackedBuf, p, ThreadPool);
    case tAVX512_VNNI::ID:
      return BTLAGemmPackQBT<tAVX512_VNNI>(QB, scales, zero_points, PackedBuf, p, ThreadPool);
    case tAVX512_VNNI_KBlock::ID:
      return BTLAGemmPackQBT<tAVX512_VNNI_KBlock>(QB, scales, zero_points, PackedBuf, p, ThreadPool);
    case tAVX_VNNI::ID:
      return BTLAGemmPackQBT<tAVX_VNNI>(QB, scales, zero_points, PackedBuf, p, ThreadPool);
    case tAVX_VNNI_KBlock::ID:
      return BTLAGemmPackQBT<tAVX_VNNI_KBlock>(QB, scales, zero_points, PackedBuf, p, ThreadPool);
    case tAMX_BF16::ID:
      return BTLAGemmPackQBT<tAMX_BF16>(QB, scales, zero_points, PackedBuf, p, ThreadPool);
    case tAMX_FP16::ID:
      return BTLAGemmPackQBT<tAMX_FP16>(QB, scales, zero_points, PackedBuf, p, ThreadPool);
    case tAVX512F::ID:
      return BTLAGemmPackQBT<tAVX512F>(QB, scales, zero_points, PackedBuf, p, ThreadPool);
    case tAVX2::ID:
      return BTLAGemmPackQBT<tAVX2>(QB, scales, zero_points, PackedBuf, p, ThreadPool);
    default:
      break;
  }
}

static inline void BTLAGemmUnpack(void* B, float* DQB, void* ThreadPool) {
  GetCPUDevice();
  using namespace bestla;
  auto prologue = storage::CollectionParser::parse_prologue_id(B);
  if (storage::CollectionParser::valid_prologue(prologue)) {
    if (prologue == BTLA_PROLOGUEB_IDS::WeightKBlockNInteger) {
      auto stor_ptr = (int8_t*)(B);
      storage::gemm::StorageWeightNInt stor;
      auto data_ptr = stor.deserialize(stor_ptr);
      stor.set_buffers(data_ptr);
      auto& id = stor.info_.core_id_;
      if (id == tAVX512_VNNI::ID) {
        BTLAGemmUnpackQBT<tAVX512_VNNI>(&stor, DQB, ThreadPool);
      } else if (id == tAVX512_VNNI_KBlock::ID) {
        BTLAGemmUnpackQBT<tAVX512_VNNI_KBlock>(&stor, DQB, ThreadPool);
      } else if (id == tAMX_INT8_SS::ID) {
        BTLAGemmUnpackQBT<tAMX_INT8_SS>(&stor, DQB, ThreadPool);
      } else if (id == tAMX_INT8_SS_KBlock::ID) {
        BTLAGemmUnpackQBT<tAMX_INT8_SS_KBlock>(&stor, DQB, ThreadPool);
      } else if (id == tAVX_VNNI::ID) {
        BTLAGemmUnpackQBT<tAVX_VNNI>(&stor, DQB, ThreadPool);
      } else if (id == tAVX_VNNI_KBlock::ID) {
        BTLAGemmUnpackQBT<tAVX_VNNI_KBlock>(&stor, DQB, ThreadPool);
      } else if (id == tAMX_BF16::ID) {
        BTLAGemmUnpackQBT<tAMX_BF16>(&stor, DQB, ThreadPool);
      } else if (id == tAMX_FP16::ID) {
        BTLAGemmUnpackQBT<tAMX_FP16>(&stor, DQB, ThreadPool);
      } else if (id == tAVX512F::ID) {
        BTLAGemmUnpackQBT<tAVX512F>(&stor, DQB, ThreadPool);
      } else if (id == tAVX2::ID) {
        BTLAGemmUnpackQBT<tAVX2>(&stor, DQB, ThreadPool);
      }
    }
  }
}

template <class GC_T, bool PC = false>
static inline void BTLAWOQGemmFp32S8ForwardT(storage::gemm::StorageWeightNInt* B, const float* A, float* C,
                                             const float* bias, int M, int N, int K, int lda, int ldc, void* ws_ptr,
                                             parallel::IThreading* ThreadPool) {
  if constexpr (PC) {
    using Parallel = parallel::gemm::SchedulerBase<GC_T>;
    using Launcher =
        wrapper::gemm::LauncherBaseV2<GC_T, prologue_a::gemm::ActivationF32KBlockQuantize,
                                      prologue_b::gemm::WeightKBlockNInt,
                                      epilogue::gemm::PcKBlockCompInt8Epilogue<epilogue::gemm::AlphaBetaProcessFp32>>;
    utils::GemmProblem gp(1, M, N, K, B->block_);
    float alpha = 1.f, beta = bias == nullptr ? 0.f : 1.f;
    auto storA = Launcher::PrologueA::createStorage(M, K, B->block_, B->corr_.zptr());
    auto wptr = (int8_t*)ws_ptr;
    storA.assign(wptr);
    typename Launcher::Param args{gp,
                                  {A, lda, &storA},
                                  {B},
                                  {{B->corr_.sptr(), B->corr_.sdtype_, storA.template SPtr<float>(),
                                    storA.template ZPtr<uint8_t>(), (int*)B->corr_.rptr(), nullptr, nullptr, K},
                                   {C, (float*)bias, ldc, 0, alpha, beta}}};
    parallel::GemmRunWithA<Parallel, Launcher>(args, ThreadPool);
  } else {
    using Parallel = parallel::gemm::SchedulerIntKBlock<GC_T>;
    using Launcher =
        wrapper::gemm::LauncherIntKBlockV2<GC_T, prologue_a::gemm::ActivationF32KBlockQuantize,
                                           prologue_b::gemm::WeightKBlockNInt, epilogue::gemm::AlphaBetaProcessFp32>;
    utils::GemmProblem gp(1, M, N, K, B->block_);
    float alpha = 1.f, beta = bias == nullptr ? 0.f : 1.f;
    auto storA = Launcher::PrologueA::createStorage(M, K, B->block_, B->corr_.zptr());
    auto wptr = (int8_t*)ws_ptr;
    storA.assign(wptr);
    typename Launcher::Param args{gp, {A, lda, &storA}, {B}, {C, (float*)bias, ldc, 0, alpha, beta}};
    parallel::GemmRunWithA<Parallel, Launcher>(args, ThreadPool);
  }
}

template <class GC_T>
static inline void BTLAWOQGemmFp32ForwardT(storage::gemm::StorageWeightNInt* B, const float* A, float* C,
                                           const float* bias, int M, int N, int K, int lda, int ldc, void* ws_ptr,
                                           parallel::IThreading* ThreadPool) {
  using Parallel = parallel::gemm::SchedulerBase<GC_T>;
  using Launcher =
      wrapper::gemm::LauncherBaseV2<GC_T, prologue_a::gemm::ActivationConverterFp32, prologue_b::gemm::WeightKBlockNInt,
                                    epilogue::gemm::AlphaBetaProcessFp32>;
  utils::GemmProblem gp(1, M, N, K, B->block_);
  float alpha = 1.f, beta = bias == nullptr ? 0.f : 1.f;
  typename Launcher::Param args{gp, {A, lda}, {B}, {C, (float*)bias, ldc, 0, alpha, beta}};
  parallel::GemmRun<Parallel, Launcher>(args, ThreadPool);
}

template <class GC_T>
static inline void BTLAGemmNPackBForwardT(void* wsptr, const void* A, const void* B, float* C, const float* bias, int M,
                                          int N, int K, parallel::IThreading* ThreadPool) {
  using Parallel = parallel::gemm::SchedulerBase<GC_T>;
  using Launcher = wrapper::gemm::LauncherBaseV2<GC_T, prologue_a::gemm::ActivationBase, prologue_b::gemm::WeightNPack,
                                                 epilogue::gemm::AlphaBetaProcessFp32>;
  using ProB = bestla::prologue_b::gemm::WeightNPack<GC_T>;
  auto stor = ProB::create_storage(N, K);
  auto wptr = (int8_t*)wsptr;
  auto data_ptr = stor.serialize(wptr);
  stor.set_buffers(data_ptr);
  auto thd = (bestla::parallel::IThreading*)ThreadPool;
  storage::gemm::StorageWeight raw;
  raw.set_nt_data(N, K, N, utils::bestla_dtype<typename GC_T::BType>, (void*)B);
  ProB::convert(raw, stor, thd);
  utils::GemmProblem gp(1, M, N, K);
  float alpha = 1.f, beta = bias == nullptr ? 0.f : 1.f;
  typename Launcher::Param args{gp, {(typename GC_T::AType*)A, K}, {0, 0, &stor}, {C, (float*)bias, N, 0, alpha, beta}};
  parallel::GemmRun<Parallel, Launcher>(args, ThreadPool);
}

static inline void BTLAGemmNPackBForward(BTLA_DTYPE dt, const void* A, const void* B, float* C, const float* bias,
                                         int M, int N, int K, int lda, int ldc, void* wsptr,
                                         parallel::IThreading* ThreadPool) {
  auto id = BTLAGemmCoreId(dt);
  if (id == tAMX_BF16::ID) {
    BTLAGemmNPackBForwardT<tAMX_BF16>(wsptr, A, B, C, bias, M, N, K, ThreadPool);
  } else if (id == tAMX_FP16::ID) {
    BTLAGemmNPackBForwardT<tAMX_FP16>(wsptr, A, B, C, bias, M, N, K, ThreadPool);
  } else if (id == tAVX512F::ID) {
    BTLAGemmNPackBForwardT<tAVX512F>(wsptr, A, B, C, bias, M, N, K, ThreadPool);
  } else if (id == tAVX2::ID) {
    BTLAGemmNPackBForwardT<tAVX2>(wsptr, A, B, C, bias, M, N, K, ThreadPool);
  }
}

static inline size_t BTLAGemmNPackBSize(int N, int K, BTLA_DTYPE dt) {
  auto id = BTLAGemmCoreId(dt);
  if (id == tAMX_BF16::ID) {
    using ProB = prologue_b::gemm::WeightNPack<tAMX_BF16>;
    auto stor = ProB::create_storage(N, K);
    return stor.total_size();
  } else if (id == tAMX_FP16::ID) {
    using ProB = prologue_b::gemm::WeightNPack<tAMX_FP16>;
    auto stor = ProB::create_storage(N, K);
    return stor.total_size();
  } else if (id == tAVX512F::ID) {
    using ProB = prologue_b::gemm::WeightNPack<tAVX512F>;
    auto stor = ProB::create_storage(N, K);
    return stor.total_size();
  } else if (id == tAVX2::ID) {
    using ProB = prologue_b::gemm::WeightNPack<tAVX2>;
    auto stor = ProB::create_storage(N, K);
    return stor.total_size();
  }
  return 0;
}

static inline size_t BTLAWOQGemmForwardWorkspace(void* B, int M, int N, int K, int lda, int ldc) {
  GetCPUDevice();
  using namespace bestla;
  auto prologue = storage::CollectionParser::parse_prologue_id(B);
  if (storage::CollectionParser::valid_prologue(prologue)) {
    if (prologue == BTLA_PROLOGUEB_IDS::WeightKBlockNInteger) {
      storage::gemm::StorageWeightNInt stor;
      auto stor_ptr = (int8_t*)(B);
      auto data_ptr = stor.deserialize(stor_ptr);
      stor.set_buffers(data_ptr);
      auto& id = stor.info_.core_id_;
      if (id == tAVX512_VNNI_KBlock::ID) {
        auto storA = prologue_a::gemm::ActivationF32KBlockQuantize<tAVX512_VNNI_KBlock>::createStorage(
            M, K, stor.block_, stor.corr_.zptr());
        return storA.mSize;
      } else if (id == tAVX512_VNNI::ID) {
        auto storA = prologue_a::gemm::ActivationF32KBlockQuantize<tAVX512_VNNI>::createStorage(M, K, stor.block_,
                                                                                                stor.corr_.zptr());
        return storA.mSize;
      } else if (id == tAMX_INT8_SS::ID) {
        auto storA = prologue_a::gemm::ActivationF32KBlockQuantize<tAMX_INT8_SS>::createStorage(M, K, stor.block_,
                                                                                                stor.corr_.zptr());
        return storA.mSize;
      } else if (id == tAMX_INT8_SS_KBlock::ID) {
        auto storA = prologue_a::gemm::ActivationF32KBlockQuantize<tAMX_INT8_SS_KBlock>::createStorage(
            M, K, stor.block_, stor.corr_.zptr());
        return storA.mSize;
      } else if (id == tAVX_VNNI::ID) {
        auto storA = prologue_a::gemm::ActivationF32KBlockQuantize<tAVX_VNNI>::createStorage(M, K, stor.block_,
                                                                                             stor.corr_.zptr());
        return storA.mSize;
      } else if (id == tAVX_VNNI_KBlock::ID) {
        auto storA = prologue_a::gemm::ActivationF32KBlockQuantize<tAVX_VNNI_KBlock>::createStorage(M, K, stor.block_,
                                                                                                    stor.corr_.zptr());
        return storA.mSize;
      }
    }
  }
  return 0;
}

static inline void BTLAWOQGemmFp32Forward(void* B, const float* A, float* C, const float* bias, int M, int N, int K,
                                          int lda, int ldc, void* ws_ptr, parallel::IThreading* ThreadPool) {
  GetCPUDevice();
  using namespace bestla;
  auto prologue = storage::CollectionParser::parse_prologue_id(B);
  if (storage::CollectionParser::valid_prologue(prologue)) {
    if (prologue == BTLA_PROLOGUEB_IDS::WeightKBlockNInteger) {
      auto stor_ptr = (int8_t*)(B);
      storage::gemm::StorageWeightNInt stor;
      auto data_ptr = stor.deserialize(stor_ptr);
      stor.set_buffers(data_ptr);
      auto& id = stor.info_.core_id_;
      if (id == tAVX512_VNNI::ID) {
        BTLAWOQGemmFp32S8ForwardT<tAVX512_VNNI, true>(&stor, A, C, bias, M, N, K, lda, ldc, ws_ptr, ThreadPool);
      } else if (id == tAVX512_VNNI_KBlock::ID) {
        BTLAWOQGemmFp32S8ForwardT<tAVX512_VNNI_KBlock>(&stor, A, C, bias, M, N, K, lda, ldc, ws_ptr, ThreadPool);
      } else if (id == tAMX_INT8_SS::ID) {
        BTLAWOQGemmFp32S8ForwardT<tAMX_INT8_SS, true>(&stor, A, C, bias, M, N, K, lda, ldc, ws_ptr, ThreadPool);
      } else if (id == tAMX_INT8_SS_KBlock::ID) {
        BTLAWOQGemmFp32S8ForwardT<tAMX_INT8_SS_KBlock>(&stor, A, C, bias, M, N, K, lda, ldc, ws_ptr, ThreadPool);
      } else if (id == tAVX_VNNI::ID) {
        BTLAWOQGemmFp32S8ForwardT<tAVX_VNNI, true>(&stor, A, C, bias, M, N, K, lda, ldc, ws_ptr, ThreadPool);
      } else if (id == tAVX_VNNI_KBlock::ID) {
        BTLAWOQGemmFp32S8ForwardT<tAVX_VNNI_KBlock>(&stor, A, C, bias, M, N, K, lda, ldc, ws_ptr, ThreadPool);
      } else if (id == tAMX_BF16::ID) {
        BTLAWOQGemmFp32ForwardT<tAMX_BF16>(&stor, A, C, bias, M, N, K, lda, ldc, ws_ptr, ThreadPool);
      } else if (id == tAMX_FP16::ID) {
        BTLAWOQGemmFp32ForwardT<tAMX_FP16>(&stor, A, C, bias, M, N, K, lda, ldc, ws_ptr, ThreadPool);
      } else if (id == tAVX512F::ID) {
        BTLAWOQGemmFp32ForwardT<tAVX512F>(&stor, A, C, bias, M, N, K, lda, ldc, ws_ptr, ThreadPool);
      } else if (id == tAVX2::ID) {
        BTLAWOQGemmFp32ForwardT<tAVX2>(&stor, A, C, bias, M, N, K, lda, ldc, ws_ptr, ThreadPool);
      }
    }
  }
}

}  // namespace bestla

namespace ark {

class CpuWrapper {
 public:
  static bestla::parallel::IThreading* get_threading() {
    static bestla::parallel::OMPThreading DefaultThreading;
    return &DefaultThreading;
  }

  static inline size_t get_packw_size(QuantParam* p) {
    auto coreid = bestla::BTLAGemmCoreId(p);
    auto size = bestla::BTLAGemmPackBSize(coreid, p);
    return size;
  }

  static void packq(const int8_t* raws8, const float* scaleptr, const int8_t* zp, int8_t* blob, QuantParam* p) {
    auto coreid = bestla::BTLAGemmCoreId(p);
    bestla::BTLAGemmPackQB(blob, raws8, scaleptr, p->asym ? zp : nullptr, coreid, p, get_threading());
  }

  static void unpackq(BTLA_DTYPE outt, int8_t* blob, void* optr, QuantParam* p) {
    assert(outt == BTLA_DTYPE::F32);
    bestla::BTLAGemmUnpack(blob, (float*)optr, get_threading());
  }

  static void gemm(int m, int n, int k, const void* A, BTLA_DTYPE dt, const void* B, bool BT, float* C,
                   const float* bias) {
    auto bsize = bestla::BTLAGemmNPackBSize(n, k, dt);
    auto wssize = bsize;
    wssize += BT ? n * k * bestla::utils::bestla_dtype_bytes(dt) : 0;
    auto ptr = DnnlContext::Instance()->get_scratch_mem(wssize, 0, nullptr);
    auto packwptr = ptr;
    auto bntptr = (int8_t*)packwptr + bsize;
    auto bptr = B;
    auto thd = get_threading();
    if (BT) {
      bestla::prologue_b::gemm::transpose(n, k, (const int8_t*)B, dt, k, bntptr, n, thd);
      bptr = bntptr;
    }
    bestla::BTLAGemmNPackBForward(dt, A, bptr, C, bias, m, n, k, k, n, packwptr, thd);
  }

  static void woq_gemm(int m, const void* a, const void* b, void* c, const void* bias, BTLA_DTYPE acdt, QuantParam* p) {
    auto wssize = bestla::BTLAWOQGemmForwardWorkspace((void*)b, m, p->n, p->k, p->k, p->n);
    auto ptr = DnnlContext::Instance()->get_scratch_mem(wssize, 0, nullptr);
    bestla::BTLAWOQGemmFp32Forward((void*)b, (const float*)a, (float*)c, (const float*)bias, m, p->n, p->k, p->k, p->n,
                                   ptr, get_threading());
  }
};

}  // namespace ark