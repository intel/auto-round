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
#include <cassert>
#include "bestla_utils.h"
#include "bestla_storage.h"
#include "bestla_device.h"
#include "bestla_parallel.h"
#include "kernel_wrapper.h"

namespace bestla {
namespace prologue_b {
namespace gemm {

template <typename WT>
static inline void transposeWeight(const int Row, const int Col, const WT* src, const int ld_src, WT* dst,
                                   const int ld_dst, parallel::IThreading* threading) {
  bestla::parallel::Scheduler2D _para;
  _para.update({threading->num_threads(), Row, Col, 16, 16});
  threading->parallel_for([&](int tidx) {
    bestla::parallel::ThreadProblem2D thdp{tidx};
    _para.getIndex(thdp);
    if (thdp.valid) {
      kernel::wrapper::Transpose2D<WT>::forward_auto(src + thdp.loc[0] * ld_src + thdp.loc[1],
                                                     dst + thdp.loc[0] + thdp.loc[1] * ld_dst, thdp.size[0],
                                                     thdp.size[1], ld_src, ld_dst);
    }
  });
}

static inline void transpose(int Row, int Col, const int8_t* src, BTLA_DTYPE dtype, int ld_src, int8_t* dst, int ld_dst,
                             parallel::IThreading* threading) {
  bestla::parallel::Scheduler2D _para;
  _para.update({threading->num_threads(), Row, Col, 16, 16});
  threading->parallel_for([&](int tidx) {
    bestla::parallel::ThreadProblem2D thdp{tidx};
    auto byte = utils::bestla_dtype_bytes(dtype);
    _para.getIndex(thdp);
    if (thdp.valid) {
      if (byte == 1) {
        kernel::wrapper::Transpose2D<int8_t>::forward_auto(src + thdp.loc[0] * ld_src + thdp.loc[1],
                                                           dst + thdp.loc[0] + thdp.loc[1] * ld_dst, thdp.size[0],
                                                           thdp.size[1], ld_src, ld_dst);
      }
      if (byte == 2) {
        kernel::wrapper::Transpose2D<int16_t>::forward_auto((int16_t*)src + thdp.loc[0] * ld_src + thdp.loc[1],
                                                            (int16_t*)dst + thdp.loc[0] + thdp.loc[1] * ld_dst,
                                                            thdp.size[0], thdp.size[1], ld_src, ld_dst);
      }
      if (byte == 4) {
        kernel::wrapper::Transpose2D<int32_t>::forward_auto((int32_t*)src + thdp.loc[0] * ld_src + thdp.loc[1],
                                                            (int32_t*)dst + thdp.loc[0] + thdp.loc[1] * ld_dst,
                                                            thdp.size[0], thdp.size[1], ld_src, ld_dst);
      }
    }
  });
}

template <typename WType>
struct ParamWeightPack {
  const WType* B;
  const int ldb;
  storage::gemm::StoragePackedWeight* packedW;
};

template <class _GemmCore_T>
class WeightPack {
 public:
  using WType = typename _GemmCore_T::BType;
  using StorageType = storage::gemm::StoragePackedWeight;
  using Param = ParamWeightPack<WType>;

  AUTOCALL StorageType createStorage(int n, int k, storage::gemm::Layout _layout = storage::gemm::Layout::NPack) {
    int KPad = k, NPad = n;
    switch (_layout) {
      case bestla::storage::gemm::Layout::NPack: {
        KPad = utils::padto(k, _GemmCore_T::KTILE);
        NPad = utils::padto(n, _GemmCore_T::NTILE);
      } break;
      case bestla::storage::gemm::Layout::T:
      case bestla::storage::gemm::Layout::NT:
      default:
        break;
    }
    StorageType tmp(_GemmCore_T::ID);
    tmp.resize(NPad, KPad, n, k, utils::bestla_dtype<WType>, _layout);
    return tmp;
  }

  AUTOCALL void packWeightTranspose(const int N, const int K, const Param& _param, parallel::IThreading* threading) {
    auto B_NT = utils::amalloc<WType>(static_cast<size_t>(N) * K);
    transposeWeight<WType>(N, K, _param.B, _param.ldb, B_NT, N, threading);
    packWeight(N, K, {B_NT, N, _param.packedW}, threading);
    utils::afree(B_NT);
  }

  // from KxN int8 symmetric weight to packed N//NtilexKPadxNTile int4 weight
  AUTOCALL void packWeight(const int N, const int K, const Param& _param, parallel::IThreading* threading) {
    parallel::Scheduler2D _para({threading->num_threads(), K, N, _GemmCore_T::KTILE, _GemmCore_T::NTILE});
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp{tidx};
      _para.getIndex(thdp);
      if (thdp.valid) {
        run(_param, thdp);
      }
    });
  }

  AUTOCALL void unpackWeight(const int N, const int K, StorageType* stor, WType* B, const int ldb,
                             parallel::IThreading* threading) {
    parallel::Scheduler2D _para({threading->num_threads(), K, N, _GemmCore_T::KTILE, _GemmCore_T::NTILE});
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp{tidx};
      _para.getIndex(thdp);
      if (thdp.valid) {
        auto rowpad = utils::padto(thdp.size[0], _GemmCore_T::KTILE);
        auto colpad = utils::padto(thdp.size[1], _GemmCore_T::NTILE);
        auto dequant = utils::amalloc<WType>((size_t)rowpad * colpad);
        auto dstptr = dequant;
        int dststep = 0;
        size_t constexpr CacheSize = size_t(100) << 10;
        int8_t tmpcache[CacheSize];
        GetCPUDevice();
        if (_cd->AVX512F()) {
          getWeight<BTLA_ISA::AVX512F>(&dstptr, &dststep, rowpad, colpad, thdp.loc[0], thdp.loc[1], {nullptr, 0, stor},
                                       tmpcache, CacheSize);
        } else if (_cd->AVX2()) {
          getWeight<BTLA_ISA::AVX2>(&dstptr, &dststep, rowpad, colpad, thdp.loc[0], thdp.loc[1], {nullptr, 0, stor},
                                    tmpcache, CacheSize);
        } else {
          getWeight<BTLA_ISA::NoSIMD>(&dstptr, &dststep, rowpad, colpad, thdp.loc[0], thdp.loc[1], {nullptr, 0, stor},
                                      tmpcache, CacheSize);
        }
        kernel::wrapper::RevertPaddingInterleaveMN<_GemmCore_T::NTILE, _GemmCore_T::PACK_ROW, WType>::forward_auto(
            dstptr, B + thdp.loc[0] * ldb + thdp.loc[1], thdp.size[0], thdp.size[1], rowpad, colpad, dststep, ldb);
        utils::afree(dequant);
      }
    });
  }

  AUTOCALL void run(const Param& _param, parallel::ThreadProblem2D& thdp) {
    auto packedw = _param.packedW;
    auto rowpadded = utils::padto(thdp.size[0], _GemmCore_T::KTILE);
    auto colpadded = utils::padto(thdp.size[1], _GemmCore_T::NTILE);
    const auto src = _param.B + thdp.loc[0] * _param.ldb + thdp.loc[1];
    const auto dst = packedw->template WPtr<WType>() + thdp.loc[0] * _GemmCore_T::NTILE + thdp.loc[1] * packedw->mKPad;
    using PaddingInterleaveMNWType =
        kernel::wrapper::PaddingInterleaveMN<_GemmCore_T::NTILE, _GemmCore_T::PACK_ROW, WType>;
    auto ret = PaddingInterleaveMNWType::forward_auto(  //
        src, dst, thdp.size[0], thdp.size[1], rowpadded, colpadded, _param.ldb, packedw->mKPad);
    _param.packedW->mLayout = storage::gemm::Layout::NPack;
    assert(ret == BTLA_CODE::Success);
    (void)ret;
  }

  TLACALL BTLA_CODE getWeight(WType** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                              const Param param, void* tmpcache, size_t cachesize) {
    auto wptr = param.packedW;
    auto KPad = wptr->mKPad;
    auto bptr = wptr->template WPtr<WType>() + n_offset * KPad + k_offset * _GemmCore_T::NTILE;
    kernel::wrapper::Memcpy2D::template forward<BTLA_ISA::NoSIMD, WType, WType>(
        bptr, *dstptr, n_size / _GemmCore_T::NTILE, _GemmCore_T::NTILE * k_size, _GemmCore_T::NTILE * KPad,
        _GemmCore_T::NTILE * k_size);
    *dststep = k_size;
    return BTLA_CODE::Success;
  }
};

template <typename _Wt, class _St>
struct ParamWeightNPack {
  const _Wt* B_;
  const int ldb_;
  const _St* stor_;
};

template <class _GemmCore_T>
class WeightNPack {
 public:
  using WType = typename _GemmCore_T::BType;
  using StorageType = storage::gemm::StorageWeight;
  using Param = ParamWeightNPack<WType, StorageType>;

  AUTOCALL StorageType create_storage(int n, int k, storage::gemm::Layout _layout = storage::gemm::Layout::NPack) {
    int KPad = k, NPad = n;
    switch (_layout) {
      case bestla::storage::gemm::Layout::NPack: {
        KPad = utils::padto(k, _GemmCore_T::KTILE);
        NPad = utils::padto(n, _GemmCore_T::NTILE);
      } break;
      case bestla::storage::gemm::Layout::T:
      case bestla::storage::gemm::Layout::NT:
      default:
        break;
    }
    StorageType tmp(_GemmCore_T::ID);
    tmp.resize(NPad, KPad, n, k, utils::bestla_dtype<WType>, _layout);
    return tmp;
  }

  AUTOCALL void unpack_weight(const int N, const int K, StorageType* stor, WType* B, const int ldb,
                              parallel::IThreading* threading) {
    parallel::Scheduler2D _para({threading->num_threads(), K, N, _GemmCore_T::KTILE, _GemmCore_T::NTILE});
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp{tidx};
      _para.getIndex(thdp);
      if (thdp.valid) {
        auto rowpad = utils::padto(thdp.size[0], _GemmCore_T::KTILE);
        auto colpad = utils::padto(thdp.size[1], _GemmCore_T::NTILE);
        auto dequant = utils::amalloc<WType>((size_t)rowpad * colpad);
        auto dstptr = dequant;
        int dststep = 0;
        size_t constexpr CacheSize = size_t(100) << 10;
        int8_t tmpcache[CacheSize];
        GetCPUDevice();
        if (_cd->AVX512F()) {
          get_weight<BTLA_ISA::AVX512F>(&dstptr, &dststep, rowpad, colpad, thdp.loc[0], thdp.loc[1], {nullptr, 0, stor},
                                        tmpcache, CacheSize);
        } else if (_cd->AVX2()) {
          get_weight<BTLA_ISA::AVX2>(&dstptr, &dststep, rowpad, colpad, thdp.loc[0], thdp.loc[1], {nullptr, 0, stor},
                                     tmpcache, CacheSize);
        } else {
          get_weight<BTLA_ISA::NoSIMD>(&dstptr, &dststep, rowpad, colpad, thdp.loc[0], thdp.loc[1], {nullptr, 0, stor},
                                       tmpcache, CacheSize);
        }
        kernel::wrapper::RevertPaddingInterleaveMN<_GemmCore_T::NTILE, _GemmCore_T::PACK_ROW, WType>::forward_auto(
            dstptr, B + thdp.loc[0] * ldb + thdp.loc[1], thdp.size[0], thdp.size[1], rowpad, colpad, dststep, ldb);
        utils::afree(dequant);
      }
    });
  }

  using PaddingInterleaveMNWType =
      kernel::wrapper::PaddingInterleaveMN<_GemmCore_T::NTILE, _GemmCore_T::PACK_ROW, WType>;

  AUTOCALL void convert(const storage::gemm::StorageWeight& src, const storage::gemm::StorageWeight& dst,
                        parallel::IThreading* threading) {
    assert(src.info_.dtype_ == utils::bestla_dtype<WType>);  // No type conversion
    assert(dst.info_.dtype_ == utils::bestla_dtype<WType>);  // No type conversion
    using storage::gemm::Layout;
    if (src.info_.layout_ == storage::gemm::Layout::NT) {
      convert_NT(src, dst, threading);
    }
    if (src.info_.layout_ == storage::gemm::Layout::T) {
      auto tmpptr = utils::amalloc<int8_t>(src.buf_size_);
      storage::gemm::StorageWeight nt_stor;
      nt_stor.set_nt_data(src.info_.n_, src.info_.k_, src.info_.n_, src.info_.dtype_, tmpptr);
      convert_T(src, nt_stor, threading);
      convert_NT(nt_stor, dst, threading);
      utils::afree(tmpptr);
    }
    if (src.info_.layout_ == storage::gemm::Layout::NPack) {
      convert_NPack(src, dst, threading);
    }
  }

  TLACALL BTLA_CODE get_weight(WType** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                               const Param param, void* tmpcache, size_t cachesize) {
    auto wptr = param.stor_;
    auto KPad = wptr->info_.kpad_;
    auto bptr = (WType*)wptr->ptr() + n_offset * KPad + k_offset * _GemmCore_T::NTILE;
    auto bptr_align = utils::cpu_pointer_align<WType>(bptr);
    if (bptr_align == bptr) {
      *dstptr = bptr;
      *dststep = KPad;
    } else {
      kernel::wrapper::Memcpy2D::template forward<BTLA_ISA::NoSIMD, WType, WType>(
          bptr, *dstptr, n_size / _GemmCore_T::NTILE, _GemmCore_T::NTILE * k_size, _GemmCore_T::NTILE * KPad,
          _GemmCore_T::NTILE * k_size);
      *dststep = k_size;
    }
    return BTLA_CODE::Success;
  }

  AUTOCALL BTLA_CODE get_weight_auto(WType** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                     const Param param, void* tmpcache, size_t cachesize) {
    GetCPUDevice();
    if (_cd->AVX512F()) {
      return get_weight<BTLA_ISA::AVX512F>(dstptr, dststep, k_size, n_size, k_offset, n_offset, param, tmpcache,
                                           cachesize);
    } else if (_cd->AVX2()) {
      return get_weight<BTLA_ISA::AVX2>(dstptr, dststep, k_size, n_size, k_offset, n_offset, param, tmpcache,
                                        cachesize);
    } else {
      return get_weight<BTLA_ISA::NoSIMD>(dstptr, dststep, k_size, n_size, k_offset, n_offset, param, tmpcache,
                                          cachesize);
    }
    return BTLA_CODE::NotSupport;
  }
  AUTOCALL void convert_NT(const storage::gemm::StorageWeight& src, const storage::gemm::StorageWeight& dst,
                           parallel::IThreading* threading) {
    assert(src.info_.dtype_ == utils::bestla_dtype<WType>);  // No type conversion
    assert(dst.info_.dtype_ == utils::bestla_dtype<WType>);  // No type conversion
    assert(src.info_.layout_ == storage::gemm::Layout::NT);
    using storage::gemm::Layout;
    if (dst.info_.layout_ == storage::gemm::Layout::NPack) {
      parallel::Scheduler2D _para(
          {threading->num_threads(), src.info_.k_, src.info_.n_, _GemmCore_T::KTILE, _GemmCore_T::NTILE});
      threading->parallel_for([&](int tidx) {
        parallel::ThreadProblem2D thdp{tidx};
        _para.getIndex(thdp);
        if (thdp.valid) {
          auto sptr = (const WType*)src.ptr() + (thdp.loc[0] * src.info_.npad_ + thdp.loc[1]);
          auto dptr = (WType*)dst.ptr() + (thdp.loc[0] * _GemmCore_T::NTILE + thdp.loc[1] * dst.info_.kpad_);
          auto rowpadded = utils::padto(thdp.size[0], _GemmCore_T::KTILE);
          auto colpadded = utils::padto(thdp.size[1], _GemmCore_T::NTILE);
          auto ret = PaddingInterleaveMNWType::forward_auto(  //
              sptr, dptr, thdp.size[0], thdp.size[1], rowpadded, colpadded, src.info_.npad_, dst.info_.kpad_);
          assert(ret == BTLA_CODE::Success);
        }
      });
    }

    if (dst.info_.layout_ == storage::gemm::Layout::T) {
      transposeWeight<WType>(src.info_.k_, src.info_.n_, (const WType*)src.buf_.buf_ptr_, src.info_.npad_,
                             (WType*)dst.buf_.buf_ptr_, dst.info_.kpad_, threading);
    }
  }

  AUTOCALL void convert_NPack(const storage::gemm::StorageWeight& src, const storage::gemm::StorageWeight& dst,
                              parallel::IThreading* threading) {
    using storage::gemm::Layout;
    assert(src.info_.dtype_ == utils::bestla_dtype<WType>);  // No type conversion
    assert(dst.info_.dtype_ == utils::bestla_dtype<WType>);  // No type conversion
    assert(src.info_.layout_ == storage::gemm::Layout::NPack);
    if (dst.info_.layout_ == storage::gemm::Layout::NPack) {
      if (dst.info_.core_id_ == src.info_.core_id_) {
        std::memcpy(dst.buf_.buf_ptr_, src.buf_.buf_ptr_, src.buf_size_);
        return;
      }
      // reorder for different Cores
    }
    if (dst.info_.layout_ == Layout::NT || dst.info_.layout_ == Layout::T) {
      auto ntptr = (WType*)dst.buf_.buf_ptr_;
      auto nt_ld = dst.info_.npad_;
      if (dst.info_.layout_ == Layout::T) {
        ntptr = (WType*)utils::amalloc<int8_t>(dst.buf_size_);
      }
      parallel::Scheduler2D _para(
          {threading->num_threads(), src.info_.k_, src.info_.n_, _GemmCore_T::KTILE, _GemmCore_T::NTILE});
      threading->parallel_for([&](int tidx) {
        parallel::ThreadProblem2D thdp{tidx};
        _para.getIndex(thdp);
        if (thdp.valid) {
          auto rowpad = utils::padto(thdp.size[0], _GemmCore_T::KTILE);
          auto colpad = utils::padto(thdp.size[1], _GemmCore_T::NTILE);
          auto dequant = utils::amalloc<WType>((size_t)rowpad * colpad);
          auto dstptr = dequant;
          int dststep = 0;
          size_t constexpr CacheSize = size_t(100) << 10;
          int8_t tmpcache[CacheSize];
          get_weight_auto(&dstptr, &dststep, rowpad, colpad, thdp.loc[0], thdp.loc[1], {nullptr, 0, &src}, tmpcache,
                          CacheSize);
          kernel::wrapper::RevertPaddingInterleaveMN<_GemmCore_T::NTILE, _GemmCore_T::PACK_ROW, WType>::forward_auto(
              dstptr, ntptr + thdp.loc[0] * nt_ld + thdp.loc[1], thdp.size[0], thdp.size[1], rowpad, colpad, dststep,
              nt_ld);
          utils::afree(dequant);
        }
      });
      if (dst.info_.layout_ == Layout::T) {
        storage::gemm::StorageWeight nt_stor;
        nt_stor.set_nt_data(src.info_.n_, src.info_.k_, src.info_.n_, src.info_.dtype_, ntptr);
        convert_NT(nt_stor, dst, threading);
        utils::afree(ntptr);
      }
    }
  }

  AUTOCALL void convert_T(const storage::gemm::StorageWeight& src, const storage::gemm::StorageWeight& dst,
                          parallel::IThreading* threading) {
    assert(src.info_.dtype_ == utils::bestla_dtype<WType>);  // No type conversion
    assert(dst.info_.dtype_ == utils::bestla_dtype<WType>);  // No type conversion
    assert(src.info_.layout_ == storage::gemm::Layout::T);
    using storage::gemm::Layout;
    if (dst.info_.layout_ == storage::gemm::Layout::NT) {
      transposeWeight<WType>(src.info_.n_, src.info_.k_, (const WType*)src.buf_.buf_ptr_, src.info_.kpad_,
                             (WType*)dst.buf_.buf_ptr_, dst.info_.npad_, threading);
    }
  }
};

struct ParamWeightKBlockNInteger {
  storage::gemm::StorageWeightKBlockNInteger* packedW;
};

template <class _GemmCore_T>
class WeightKBlockNInteger {
 public:
  using StorageWeight = storage::gemm::StorageWeightKBlockNInteger;
  using BType = typename _GemmCore_T::BType;
  using Param = ParamWeightKBlockNInteger;

  AUTOCALL StorageWeight createStorage(int n, int k, int blocksize, BTLA_DTYPE qtype, BTLA_DTYPE scat, bool is_asym) {
    int KPad = utils::padto(k, _GemmCore_T::KTILE);
    int NPad = utils::padto(n, _GemmCore_T::NTILE);
    StorageWeight tmp(_GemmCore_T::ID);
    tmp.resize(NPad, KPad, blocksize <= 0 ? KPad : blocksize, n, k, qtype, scat, is_asym);
    return tmp;
  }

  AUTOCALL void convertTransStorage(StorageWeight& srcstor, StorageWeight& dststor, parallel::IThreading* threading) {
    auto s8buf = utils::amalloc<int8_t>((size_t)srcstor.mK * srcstor.mN);
    auto s8transbuf = utils::amalloc<int8_t>((size_t)srcstor.mKPad * srcstor.mNPad);
    unpackWeight(srcstor.mN, srcstor.mK, &srcstor, s8buf, srcstor.mN, threading);
    transposeWeight<int8_t>(srcstor.mK, srcstor.mN, s8buf, srcstor.mN, s8transbuf, srcstor.mKPad, threading);
    compressWeight(srcstor.mKPad, srcstor.mNPad, s8transbuf, srcstor.mKPad, dststor.WPtr<int8_t>(), srcstor.mDType,
                   threading);
    if (s8buf) {
      utils::afree(s8buf);
    }
    if (s8transbuf) {
      utils::afree(s8transbuf);
    }
    int nk_scale = utils::updiv(srcstor.mKPad, srcstor.mBlockSize);
    if (srcstor.mCorrection.mScaEleSize == 4) {
      transposeWeight<float>(nk_scale, srcstor.mNPad, srcstor.template SPtr<float>(), srcstor.mNPad,
                             dststor.template SPtr<float>(), dststor.CStep(), threading);
    } else if (srcstor.mCorrection.mScaEleSize == 2) {
      transposeWeight<uint16_t>(nk_scale, srcstor.mNPad, srcstor.template SPtr<uint16_t>(), srcstor.mNPad,
                                dststor.template SPtr<uint16_t>(), dststor.CStep(), threading);
    }
  }
  AUTOCALL void doubleQuantScale(float* scale, size_t scale_size, int dq_blocksize, BTLA_DTYPE qtype,
                                 utils::aligned_vector<float>* dq_buf) {
    if (qtype == BTLA_DTYPE::DQ8_BNB) {
      dq_buf->resize(utils::updiv(scale_size, dq_blocksize) + 1);  // add 1 for offset.
      kernel::ref::dq8_bnb_double_quant<false>(scale, scale_size, dq_blocksize, dq_buf->data());
    } else {
      assert(0);
    }
  }

  AUTOCALL void setDoubleQuantCorrection(utils::avector<float>* dq_buf, StorageWeight* ptr) {
    if (ptr->SDtype() == BTLA_DTYPE::DQ8_BNB) {
      auto packw_dqbuf_ptr = ptr->DQPtr<float>();
      memcpy(packw_dqbuf_ptr, dq_buf->data(), dq_buf->size() * sizeof(float));
    } else {
      assert(0);
    }
  }

  AUTOCALL void enableShuffle(StorageWeight* stor) { stor->enable_shuffle(); }

  AUTOCALL void setDoubleQuantBlkSize(StorageWeight* stor, BTLA_DTYPE stype, int dq_blksize) {
    stor->mDqBlockSize = dq_blksize;
    auto nk_scale = utils::updiv(stor->mKPad, stor->mBlockSize);
    if (stor->IsAsym() || dq_blksize % 8 != 0) assert(0);
    stor->mCorrection.enable_double_quant(utils::updiv(nk_scale * stor->mN, dq_blksize), stype);
    stor->update_size();
  }

  AUTOCALL void packTransposeWeight(const int N, const int K, const float* B, const int ldb, StorageWeight* stor,
                                    parallel::IThreading* threading) {
    auto B_NT = utils::amalloc<float>(static_cast<size_t>(N) * K);
    transposeWeight<float>(N, K, B, ldb, B_NT, N, threading);
    packWeight(N, K, B_NT, N, stor, threading);
    utils::afree(B_NT);
  }

  // from packed N//NtilexKPadxNTile int8 weight to KxN f32 weight
  AUTOCALL void unpackTransposeWeight(const int N, const int K, StorageWeight* stor, float* B, const int ldb,
                                      parallel::IThreading* threading) {
    auto B_NT = utils::amalloc<float>(static_cast<size_t>(N) * K);
    unpackWeight(N, K, stor, B_NT, N, threading);
    transposeWeight<float>(K, N, B_NT, N, B, ldb, threading);
    utils::afree(B_NT);
  }

  // from KxN f32 weight to packed N//NtilexKPadxNTile int8 weight
  AUTOCALL void packWeight(const int N, const int K, const float* B, const int ldb, StorageWeight* ptr,
                           parallel::IThreading* threading) {
    auto tmpq = utils::amalloc<int8_t>(static_cast<size_t>(N) * K);
    int nk_scale = utils::updiv(K, ptr->mBlockSize);
    auto ssize = static_cast<size_t>(N) * nk_scale;
    auto Tscales = utils::amalloc<float>(ssize);
    auto Tzps = utils::amalloc<int8_t>(ptr->IsAsym() ? ssize : 0);
    quantizeWeight(N, K, B, ldb, tmpq, Tscales, Tzps, ptr, threading);
    packQWeight(N, K, tmpq, N, Tscales, Tzps, ptr, threading);
    utils::afree(tmpq);
    utils::afree(Tscales);
    utils::afree(Tzps);
  }
  template <typename T>
  AUTOCALL void unpackWeight(const int N, const int K, StorageWeight* stor, T* B, const int ldb,
                             parallel::IThreading* threading) {
    parallel::Scheduler2D _para({threading->num_threads(), K, N, _GemmCore_T::KTILE, _GemmCore_T::NTILE});
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp{tidx};
      _para.getIndex(thdp);
      if (thdp.valid) {
        auto rowpad = utils::padto(thdp.size[0], _GemmCore_T::KTILE);
        auto colpad = utils::padto(thdp.size[1], _GemmCore_T::NTILE);
        auto dequant = utils::amalloc<T>((size_t)rowpad * colpad);
        auto dstptr = dequant;
        int dststep = 0;
        size_t constexpr CacheSize = size_t(100) << 10;
        int8_t tmpcache[CacheSize];
        GetCPUDevice();
        if (_cd->AVX512F()) {
          getWeight<BTLA_ISA::AVX512F>(&dstptr, &dststep, rowpad, colpad, thdp.loc[0], thdp.loc[1], {stor}, tmpcache,
                                       CacheSize);
        } else if (_cd->AVX2()) {
          getWeight<BTLA_ISA::AVX2>(&dstptr, &dststep, rowpad, colpad, thdp.loc[0], thdp.loc[1], {stor}, tmpcache,
                                    CacheSize);
        } else {
          getWeight<BTLA_ISA::NoSIMD>(&dstptr, &dststep, rowpad, colpad, thdp.loc[0], thdp.loc[1], {stor}, tmpcache,
                                      CacheSize);
        }
        kernel::wrapper::RevertPaddingInterleaveMN<_GemmCore_T::NTILE, _GemmCore_T::PACK_ROW, T>::forward_auto(
            dstptr, B + thdp.loc[0] * ldb + thdp.loc[1], thdp.size[0], thdp.size[1], rowpad, colpad, dststep, ldb);
        utils::afree(dequant);
      }
    });
  }

  AUTOCALL void setQuantCorrection(const int N, const int K, const int8_t* zero_points, const float* scales,
                                   StorageWeight* stor, parallel::IThreading* threading) {
    int rawnk_scale = utils::updiv(K, stor->mBlockSize);
    int nk_scale = utils::updiv(stor->mKPad, stor->mBlockSize);
    parallel::Scheduler2D _para({threading->num_threads(), 1, nk_scale, 1, 1});
    if (stor->SDtype() == BTLA_DTYPE::BF16 || stor->SDtype() == BTLA_DTYPE::F16 || stor->SDtype() == BTLA_DTYPE::F32) {
      threading->parallel_for([&](int tidx) {
        parallel::ThreadProblem2D thdp{tidx};
        _para.getIndex(thdp);
        if (thdp.valid) {
          int rows = thdp.loc[1] + thdp.size[1] <= rawnk_scale ? thdp.size[1] : rawnk_scale - thdp.loc[1];
          if (scales) {
            if (stor->SDtype() == BTLA_DTYPE::BF16) {
              kernel::wrapper::Memcpy2DFp32TPadding<utils::bf16>::forward_auto(
                  scales + thdp.loc[1] * N, stor->template SPtr<utils::bf16>() + thdp.loc[1] * stor->mNPad, rows, N,
                  N * sizeof(scales[0]), stor->mNPad * sizeof(utils::bf16), true);
            } else if (stor->SDtype() == BTLA_DTYPE::F32) {
              kernel::wrapper::Memcpy2DPadding::forward(
                  scales + thdp.loc[1] * N, stor->template SPtr<float>() + thdp.loc[1] * stor->mNPad, rows,
                  N * sizeof(float), N * sizeof(scales[0]), stor->mNPad * sizeof(float), true);
            } else if (stor->SDtype() == BTLA_DTYPE::F16) {
              kernel::wrapper::Memcpy2DFp32TPadding<utils::fp16>::forward_auto(
                  scales + thdp.loc[1] * N, stor->template SPtr<utils::fp16>() + thdp.loc[1] * stor->mNPad, rows, N,
                  N * sizeof(scales[0]), stor->mNPad * sizeof(utils::fp16), true);
            }
            if (rows < thdp.size[1]) {
              auto sb = bestla::utils::bestla_dtype_bytes(stor->SDtype());
              if (sb == 2) {
                std::memset(stor->template SPtr<utils::fp16>() + (thdp.loc[1] + rows) * stor->mNPad, 0,
                            sb * (thdp.size[1] - rows) * stor->mNPad);
              } else if (sb == 4) {
                std::memset(stor->template SPtr<float>() + (thdp.loc[1] + rows) * stor->mNPad, 0,
                            sb * (thdp.size[1] - rows) * stor->mNPad);
              } else {
                assert(0);
              }
            }
          }
          if (zero_points) {
            kernel::wrapper::Memcpy2DPadding::forward(
                zero_points + thdp.loc[1] * N, stor->template ZPtr<int8_t>() + thdp.loc[1] * stor->mNPad, rows,
                N * sizeof(zero_points[0]), N * sizeof(zero_points[0]), sizeof(int8_t) * stor->mNPad, true);

            if (rows < thdp.size[1]) {
              std::memset(stor->template ZPtr<int8_t>() + (thdp.loc[1] + rows) * stor->mNPad, 0,
                          sizeof(int8_t) * (thdp.size[1] - rows) * stor->mNPad);
            }
          }
        }
      });
    } else if (stor->SDtype() == BTLA_DTYPE::F8_E8M0) {
      threading->parallel_for([&](int tidx) {
        parallel::ThreadProblem2D thdp{tidx};
        _para.getIndex(thdp);
        if (thdp.valid) {
          for (int i = thdp.loc[1]; i < thdp.loc[1] + thdp.size[1]; i++) {
            if (i < rawnk_scale) {
              if (scales != nullptr) {
                for (size_t j = 0; j < N; j++) {
                  stor->template SPtr<utils::f8>()[j + i * stor->mNPad] = static_cast<int8_t>(scales[i * N + j]);
                }
              }
            } else {
              if (scales != nullptr)
                std::memset(stor->template SPtr<utils::f8>() + i * stor->mNPad, 0, stor->mNPad * sizeof(utils::f8));
            }
          }
        }
      });
    } else if (stor->SDtype() == BTLA_DTYPE::DQ8_BNB) {
      threading->parallel_for([&](int tidx) {
        parallel::ThreadProblem2D thdp{tidx};
        _para.getIndex(thdp);
        if (thdp.valid) {
          for (int i = thdp.loc[1]; i < thdp.loc[1] + thdp.size[1]; i++) {
            if (i < rawnk_scale) {
              if (scales != nullptr) {
                for (size_t j = 0; j < N; j++) {
                  stor->template SPtr<uint8_t>()[j + i * stor->mNPad] = static_cast<uint8_t>(scales[i * N + j]);
                }
              }
            } else {
              if (scales != nullptr)
                std::memset(stor->template SPtr<uint8_t>() + i * stor->mNPad, 0, stor->mNPad * sizeof(uint8_t));
            }
          }
        }
      });
    } else {
      assert(0);
    }
  }

  AUTOCALL void setShuffleIndices(const int* groupindices, StorageWeight* stor, parallel::IThreading* threading) {
    auto groupsize = utils::updiv(stor->mK, stor->mBlockSize);
    parallel::Scheduler2D _para({threading->num_threads(), 1, groupsize, 1, 1});
    auto countptr = utils::amalloc<int>(groupsize);
    std::memset(countptr, 0, groupsize * sizeof(int));
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp{tidx};
      _para.getIndex(thdp);
      if (thdp.valid) {
        auto siptr = stor->ShfIndice();
        for (int i = 0; i < stor->mK; i++) {
          if (groupindices[i] >= thdp.loc[1] && groupindices[i] < thdp.loc[1] + thdp.size[1]) {
            siptr[groupindices[i] * stor->mBlockSize + countptr[groupindices[i]]] = i;
            countptr[groupindices[i]]++;
          }
        }
      }
    });
    utils::afree(countptr);
  }

  AUTOCALL void setTransposeQuantCorrection(const int N, const int K, const int8_t* zero_pointsT, const float* scalesT,
                                            StorageWeight* stor, parallel::IThreading* threading) {
    int rawnk_scale = utils::updiv(K, stor->mBlockSize);
    auto scales = scalesT ? utils::amalloc<float>(rawnk_scale * N) : nullptr;
    auto zero_points = zero_pointsT ? utils::amalloc<int8_t>(rawnk_scale * N) : nullptr;
    if (scales) {
      transposeWeight<float>(N, rawnk_scale, scalesT, rawnk_scale, scales, N, threading);
    }
    if (zero_points) {
      transposeWeight<int8_t>(N, rawnk_scale, zero_pointsT, rawnk_scale, zero_points, N, threading);
    }
    setQuantCorrection(N, K, zero_points, scales, stor, threading);
    if (scales) {
      utils::afree(scales);
    }
    if (zero_points) {
      utils::afree(zero_points);
    }
  }

  AUTOCALL void packQWeight(const int N, const int K, const int8_t* B, const int ldb, const float* scales,
                            const int8_t* zero_points, StorageWeight* stor, parallel::IThreading* threading) {
    if (stor->SDtype() == BTLA_DTYPE::DQ8_BNB) assert(stor->mDqBlockSize != 0);
    if (stor->IsDoubleQuant()) {
      int nk_scale = utils::updiv(K, stor->mBlockSize);
      auto ssize = static_cast<size_t>(N) * nk_scale;
      utils::avector<float> dq_buf;
      doubleQuantScale(const_cast<float*>(scales), ssize, stor->mDqBlockSize, stor->SDtype(), &dq_buf);
      setDoubleQuantCorrection(&dq_buf, stor);
    }
    setQuantCorrection(N, K, zero_points, scales, stor, threading);
    if (stor->mDType == BTLA_DTYPE::S8) {
      reorderWeight(N, K, B, ldb, stor->WPtr<int8_t>(), threading);
    } else {
      auto reordered = utils::amalloc<int8_t>((size_t)stor->mKPad * stor->mNPad);
      reorderWeight(N, K, B, ldb, reordered, threading);
      compressWeight(stor->mNPad, stor->mKPad, reordered, stor->mNPad, stor->WPtr<int8_t>(), stor->mDType, threading);
      utils::afree(reordered);
    }
    reduceWeight(stor, threading);
  }

  AUTOCALL void packNbitsWeightQ4(const int N, const int K, bool isasym, const uint8_t* B, const int ldb,
                                  const float* scales, const uint8_t* zero_points, void* ptr,
                                  parallel::IThreading* threading) {
    auto stor = reinterpret_cast<StorageWeight*>(ptr);
    auto tmp = utils::amalloc<float>(static_cast<size_t>(stor->mKPad) * stor->mNPad);
    auto blks = utils::updiv(K, stor->mBlockSize);
    auto blks_padding2 = utils::padto(blks, 2);
    auto tmpscales = tmp;
    auto tmpzeropoints = reinterpret_cast<int8_t*>(tmpscales + N * blks);
    assert(isasym == (zero_points != nullptr));
    if (scales) {
      for (size_t i = 0; i < N * blks; i += 1) {
        tmpscales[i] = scales[i];
      }
    }
    if (zero_points) {
      for (size_t i = 0; i < N; i += 1) {
        for (size_t ib = 0; ib < blks; ib += 2) {
          auto tmpzp = *(zero_points + i * blks_padding2 / 2 + ib / 2);
          tmpzeropoints[i * blks + ib] = (tmpzp & 0x0f) - 8;
          if (ib + 1 < blks) {
            tmpzeropoints[i * blks + ib + 1] = ((tmpzp & 0xf0) >> 4) - 8;
          }
        }
      }
    }

    setTransposeQuantCorrection(N, K, zero_points ? tmpzeropoints : nullptr, scales ? tmpscales : nullptr, stor,
                                threading);
    if (B) {
      auto s8ptr = (int8_t*)tmp;
      auto transposeunpackfunc_u4s4 = [&]() {
        parallel::Scheduler2D para({threading->num_threads(), N, K, 1, 2});
        threading->parallel_for([&](int tid) {
          parallel::ThreadProblem2D thdp{tid};
          para.getIndex(thdp);
          if (thdp.valid) {
            for (size_t i = thdp.loc[0]; i < thdp.loc[0] + thdp.size[0]; i++) {
              for (size_t j = thdp.loc[1]; j < thdp.loc[1] + thdp.size[1]; j += 2) {
                auto src = *(B + i * ldb / 2 + j / 2);
                s8ptr[(j + 0) * N + i] = ((src & 0xf) - 8);
                s8ptr[(j + 1) * N + i] = (((src & 0xf0) >> 4) - 8);
              }
            }
          }
        });
      };
      transposeunpackfunc_u4s4();
      auto reordered = s8ptr + static_cast<size_t>(K) * N;
      reorderWeight(N, K, s8ptr, N, reordered, threading);
      compressWeight(stor->mNPad, stor->mKPad, reordered, stor->mNPad, stor->WPtr<int8_t>(), stor->mDType, threading);
    }
    utils::afree(tmp);
  }

  AUTOCALL void reduceWeight(StorageWeight* stor, parallel::IThreading* threading) {
    if (stor->HasReduce()) {
      auto deq = utils::amalloc<int8_t>((size_t)stor->mK * stor->mN);
      unpackWeight(stor->mN, stor->mK, stor, deq, stor->mN, threading);
      reduce(stor->mN, stor->mK, stor->mBlockSize, deq, stor->mN, stor->template RPtr<int32_t>(), stor->CStep(),
             threading);
      utils::afree(deq);
    }
  }

  AUTOCALL void quantizeWeight(const int N, const int K, const float* B, const int ldb, int8_t* qB, float* scales,
                               int8_t* zero_points, void* stor, parallel::IThreading* threading) {
    auto ptr = reinterpret_cast<StorageWeight*>(stor);
    int bsize = ptr->mBlockSize == -1 ? K : ptr->mBlockSize;
    parallel::Scheduler2D _para({threading->num_threads(), K, N, bsize, 16});
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp({tidx});
      _para.getIndex(thdp);
      if (thdp.valid) {
        kernel::wrapper::QuantizeSignIntRowBlock::forward_auto(
            B + thdp.loc[0] * ldb + thdp.loc[1], qB + thdp.loc[0] * N + thdp.loc[1], thdp.size[0], thdp.size[1], ldb, N,
            scales + thdp.loc[0] / bsize * N + thdp.loc[1],
            zero_points == nullptr ? zero_points : zero_points + thdp.loc[0] / bsize * N + thdp.loc[1], ptr->mBlockSize,
            ptr->mDType);
      }
    });
  }

  AUTOCALL void reorderWeight(const int N, const int K, const int8_t* B, const int ldb, int8_t* dstptr,
                              parallel::IThreading* threading) {
    int KPad = utils::padto(K, _GemmCore_T::KTILE);
    parallel::Scheduler2D _para({threading->num_threads(), K, N, _GemmCore_T::KTILE, _GemmCore_T::NTILE});
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp({tidx});
      _para.getIndex(thdp);
      if (thdp.valid) {
        auto rowpadded = utils::padto(thdp.size[0], _GemmCore_T::KTILE);
        auto colpadded = utils::padto(thdp.size[1], _GemmCore_T::NTILE);
        const auto src = B + thdp.loc[0] * ldb + thdp.loc[1];
        const auto dst = dstptr + thdp.loc[0] * _GemmCore_T::NTILE + thdp.loc[1] * KPad;
        using PaddingInterleaveMNWType =
            kernel::wrapper::PaddingInterleaveMN<_GemmCore_T::NTILE, _GemmCore_T::PACK_ROW, int8_t>;
        auto ret = PaddingInterleaveMNWType::forward_auto(  //
            src, dst, thdp.size[0], thdp.size[1], rowpadded, colpadded, ldb, KPad);
        assert(ret == BTLA_CODE::Success);
        (void)ret;
      }
    });
  }

  AUTOCALL void compressBit3Weight(const int N, const int K, const int8_t* B, int8_t* dstptr, BTLA_DTYPE qtype,
                                   parallel::IThreading* threading) {
    auto bit1_offset = size_t(N) * K;
    auto bit2ptr = reinterpret_cast<utils::bit2x4*>(dstptr);
    auto bit1ptr = reinterpret_cast<utils::bit1x8*>(dstptr + bit1_offset / 4);
    auto ret = kernel::wrapper::CompressBit3::forward_auto(B, bit2ptr, bit1ptr, bit1_offset);
    assert(ret == BTLA_CODE::Success);
  }

  AUTOCALL void compressBit5Weight(const int N, const int K, const int8_t* B, int8_t* dstptr, BTLA_DTYPE qtype,
                                   parallel::IThreading* threading) {
    auto bit1_offset = size_t(N) * K;
    auto bit4ptr = reinterpret_cast<utils::bit4x2*>(dstptr);
    auto bit1ptr = reinterpret_cast<utils::bit1x8*>(dstptr + bit1_offset / 2);
    auto ret = kernel::wrapper::CompressBit5::forward_auto(B, bit4ptr, bit1ptr, bit1_offset);
    assert(ret == BTLA_CODE::Success);
  }

  AUTOCALL void compressBit6Weight(const int N, const int K, const int8_t* B, int8_t* dstptr, BTLA_DTYPE qtype,
                                   parallel::IThreading* threading) {
    auto bit2_offset = size_t(N) * K;
    auto bit4ptr = reinterpret_cast<utils::bit4x2*>(dstptr);
    auto bit2ptr = reinterpret_cast<utils::bit2x4*>(dstptr + bit2_offset / 2);
    auto ret = kernel::wrapper::CompressBit6::forward_auto(B, bit4ptr, bit2ptr, bit2_offset);
    assert(ret == BTLA_CODE::Success);
  }

  AUTOCALL void compressBit7Weight(const int N, const int K, const int8_t* B, int8_t* dstptr, BTLA_DTYPE qtype,
                                   parallel::IThreading* threading) {
    auto eltsize = size_t(N) * K;
    auto bit4ptr = reinterpret_cast<utils::bit4x2*>(dstptr);
    auto bit2ptr = reinterpret_cast<utils::bit2x4*>(bit4ptr + eltsize / 2);
    auto bit1ptr = reinterpret_cast<utils::bit1x8*>(bit2ptr + eltsize / 4);
    auto ret = kernel::wrapper::CompressBit7::forward_auto(B, bit4ptr, bit2ptr, bit1ptr, eltsize);
    assert(ret == BTLA_CODE::Success);
  }

  AUTOCALL void compressBit2Weight(const int N, const int K, const int8_t* B, int8_t* dstptr, BTLA_DTYPE qtype,
                                   parallel::IThreading* threading) {
    // TODO(zhe): 1D parallel compress
    parallel::Scheduler2D _para({threading->num_threads(), 1, K * N, 1, 64});
    auto bit2ptr = reinterpret_cast<utils::bit2x4*>(dstptr);
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp({tidx});
      _para.getIndex(thdp);
      if (thdp.valid) {
        auto ret =
            kernel::wrapper::CompressBit2::forward_auto(B + thdp.loc[1], bit2ptr + thdp.loc[1] / 4, thdp.size[1]);
        assert(ret == BTLA_CODE::Success);
        (void)ret;
      }
    });
  }

  AUTOCALL void compressBit1Weight(const int N, const int K, const int8_t* B, int8_t* dstptr, BTLA_DTYPE qtype,
                                   parallel::IThreading* threading) {
    // TODO(zhe): 1D parallel compress
    parallel::Scheduler2D _para({threading->num_threads(), 1, K * N, 1, 64});
    auto bit1ptr = reinterpret_cast<utils::bit1x8*>(dstptr);
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp({tidx});
      _para.getIndex(thdp);
      if (thdp.valid) {
        auto ret =
            kernel::wrapper::CompressBit1::forward_auto(B + thdp.loc[1], bit1ptr + thdp.loc[1] / 8, thdp.size[1]);
        assert(ret == BTLA_CODE::Success);
        (void)ret;
      }
    });
  }

  AUTOCALL void compressBit4Weight(const int N, const int K, const int8_t* B, int8_t* dstptr, BTLA_DTYPE qtype,
                                   parallel::IThreading* threading) {
    parallel::Scheduler2D _para({threading->num_threads(), 1, K * N, 1, 64});
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp({tidx});
      _para.getIndex(thdp);
      if (thdp.valid) {
        BTLA_CODE ret = BTLA_CODE::NotSupport;
        if (qtype == BTLA_DTYPE::S4_CLIP) {
          auto bit4ptr = reinterpret_cast<utils::int4x2*>(dstptr);
          ret = kernel::wrapper::CompressS8S4::forward_auto(B + thdp.loc[1], bit4ptr + thdp.loc[1] / 2, thdp.size[1]);
        } else if (qtype == BTLA_DTYPE::F4_BNB || qtype == BTLA_DTYPE::F4_NF4 || qtype == BTLA_DTYPE::F4_E2M1) {
          auto bit4ptr = reinterpret_cast<utils::f4x2*>(dstptr);
          ret = kernel::wrapper::CompressFp4::forward_auto(B + thdp.loc[1], bit4ptr + thdp.loc[1] / 2, thdp.size[1]);
        } else {
          assert(0);
        }
        assert(ret == BTLA_CODE::Success);
        (void)ret;
      }
    });
  }

  AUTOCALL void compressWeight(const int N, const int K, const int8_t* B, const int ldb, int8_t* dstptr,
                               BTLA_DTYPE qtype, parallel::IThreading* threading) {
    if (qtype == BTLA_DTYPE::S7_CLIP) return compressBit7Weight(N, K, B, dstptr, qtype, threading);
    if (qtype == BTLA_DTYPE::S6_CLIP) return compressBit6Weight(N, K, B, dstptr, qtype, threading);
    if (qtype == BTLA_DTYPE::S5_CLIP) return compressBit5Weight(N, K, B, dstptr, qtype, threading);
    if (qtype == BTLA_DTYPE::S4_CLIP) return compressBit4Weight(N, K, B, dstptr, qtype, threading);
    if (qtype == BTLA_DTYPE::S3_CLIP) return compressBit3Weight(N, K, B, dstptr, qtype, threading);
    if (qtype == BTLA_DTYPE::S2_CLIP) return compressBit2Weight(N, K, B, dstptr, qtype, threading);
    if (qtype == BTLA_DTYPE::S1_CLIP) return compressBit1Weight(N, K, B, dstptr, qtype, threading);
    if (qtype == BTLA_DTYPE::F4_BNB || qtype == BTLA_DTYPE::F4_NF4 || qtype == BTLA_DTYPE::F4_E2M1)
      return compressBit4Weight(N, K, B, dstptr, qtype, threading);
  }

  AUTOCALL void reduce(const int N, const int K, const int KBlock, const int8_t* B, const int ldb, int32_t* rptr,
                       const int ldr, parallel::IThreading* threading) {
    parallel::Scheduler2D _para({threading->num_threads(), K, N, KBlock, 16});
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp({tidx});
      _para.getIndex(thdp);
      if (thdp.valid) {
        const auto src = B + thdp.loc[0] * ldb + thdp.loc[1];
        const auto dst = rptr + thdp.loc[1] + thdp.loc[0] / KBlock * ldr;
        using RowReduceSum = kernel::wrapper::RowReduceSum<int8_t, int32_t>;
        for (int i = 0; i < thdp.size[0]; i += KBlock) {
          int rowremain = utils::remainsize(thdp.loc[0] + i, K, KBlock);
          auto ret = RowReduceSum::forward_auto(src + i * ldb, ldb, rowremain, thdp.size[1], dst + i / KBlock * ldr);
          assert(ret == BTLA_CODE::Success);
          (void)ret;
        }
      }
    });
  }

 public:
  template <BTLA_ISA ISA_T, typename _T>
  static inline BTLA_CODE getWeight(_T** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                    const Param& _param, void* tmpcache, size_t cachesize) {
    if constexpr (std::is_same_v<_T, int8_t>) {
      return getQWeight<ISA_T>(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
    }
    return getFpWeight<ISA_T>(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
  }

  TLACALL BTLA_CODE getQWeight(int8_t** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                               const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = _param.packedW;
    if (wptr->mDType == BTLA_DTYPE::S8) {
      return getQ8Weight<ISA_T>(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
    } else if (wptr->mDType == BTLA_DTYPE::S6_CLIP) {
      return getQ6Weight<ISA_T>(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
    } else if (wptr->mDType == BTLA_DTYPE::S5_CLIP) {
      return getQ5Weight<ISA_T>(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
    } else if (wptr->mDType == BTLA_DTYPE::S4_CLIP) {
      return getQ4Weight<ISA_T>(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
    } else if (wptr->mDType == BTLA_DTYPE::S3_CLIP) {
      return getQ3Weight<ISA_T>(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
    } else if (wptr->mDType == BTLA_DTYPE::S2_CLIP) {
      return getQ2Weight<ISA_T>(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
    } else if (wptr->mDType == BTLA_DTYPE::S7_CLIP) {
      return getQ7Weight<ISA_T>(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
    } else if (wptr->mDType == BTLA_DTYPE::S1_CLIP) {
      return getQ1Weight<ISA_T>(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
    } else {
      assert(0);
    }
    return BTLA_CODE::NotSupport;
  }

  TLACALL BTLA_CODE getScale(float** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                             const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = _param.packedW;
    if (wptr->SDtype() == BTLA_DTYPE::F32) {
      auto aptr = wptr->template SPtr<float>();
      kernel::wrapper::Memcpy2D::template forward<BTLA_ISA::NoSIMD>(
          aptr + k_offset / wptr->mBlockSize * wptr->CStep() + n_offset, *dstptr,
          utils::updiv(k_size, wptr->mBlockSize), n_size, wptr->CStep(), n_size);
      *dststep = n_size;
    }
    if (wptr->SDtype() == BTLA_DTYPE::BF16) {
      auto aptr = wptr->template SPtr<utils::bf16>();
      kernel::wrapper::Memcpy2DBf16CvtFp32::forward<ISA_T>(
          aptr + k_offset / wptr->mBlockSize * wptr->CStep() + n_offset, *dstptr,
          utils::updiv(k_size, wptr->mBlockSize), n_size, wptr->CStep() * 2, n_size * 4, false);
      *dststep = n_size;
    }
    if (wptr->SDtype() == BTLA_DTYPE::F16) {
      auto aptr = wptr->template SPtr<utils::fp16>();
      kernel::wrapper::Memcpy2DFp16CvtFp32::forward<ISA_T>(
          aptr + k_offset / wptr->mBlockSize * wptr->CStep() + n_offset, *dstptr,
          utils::updiv(k_size, wptr->mBlockSize), n_size, wptr->CStep() * 2, n_size * 4, false);
      *dststep = n_size;
    }
    if (wptr->SDtype() == BTLA_DTYPE::DQ8_BNB) {
      auto aptr = wptr->template SPtr<uint8_t>();
      auto internal_k_offset = k_offset / wptr->mBlockSize;
      auto dq_offset_idx = static_cast<int>(wptr->mCorrection.mDQCorrectionBuf.mBufSize / sizeof(float) - 1);
      kernel::wrapper::Dq8GetScale::template forward<ISA_T>(
          aptr + internal_k_offset * wptr->CStep() + n_offset, *dstptr, utils::updiv(k_size, wptr->mBlockSize), n_size,
          internal_k_offset * wptr->mN + n_offset, wptr->mDqBlockSize, dq_offset_idx, wptr->DQPtr<float>(),
          wptr->CStep(), n_size, false, wptr->mN);
    }
    return BTLA_CODE::Success;
  }

  TLACALL BTLA_CODE getReduce(int32_t** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                              const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = _param.packedW;
    auto rptr = wptr->template RPtr<int32_t>();
    kernel::wrapper::Memcpy2D::template forward<BTLA_ISA::NoSIMD>(
        rptr + k_offset / wptr->mBlockSize * wptr->CStep() + n_offset, *dstptr, utils::updiv(k_size, wptr->mBlockSize),
        n_size, wptr->CStep(), n_size);
    *dststep = n_size;
    return BTLA_CODE::Success;
  }

 protected:
  template <BTLA_ISA ISA_T, typename _T>
  static inline BTLA_CODE getFpWeight(_T** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                      const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = _param.packedW;
    auto NPad = wptr->mNPad;
    auto KPad = wptr->mKPad;
    int constexpr ColSize = _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      auto zptr = wptr->template ZPtr<int8_t>();
      if (wptr->mDType == BTLA_DTYPE::S4_CLIP) {
        if (wptr->SDtype() == BTLA_DTYPE::DQ8_BNB) {
          auto internal_n_offset = n_offset + i;
          int dq_offset = static_cast<int>(wptr->mCorrection.mDQCorrectionBuf.mBufSize / sizeof(float) - 1);
          kernel::wrapper::DecompressDQKBlockS4Fp<_T, _GemmCore_T::PACK_ROW>::template forward<ISA_T,
                                                                                               BTLA_DTYPE::S4_CLIP>(
              wptr->template WPtr<utils::int4x2>() + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2 +
                  i * KPad / 2,
              *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize,
              wptr->template SPtr<uint8_t>(), wptr->template DQPtr<float>(), k_offset / _GemmCore_T::PACK_ROW,
              internal_n_offset, wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad, wptr->mN, wptr->mDqBlockSize,
              dq_offset, tmpcache, cachesize);
        } else {
          auto sptr = wptr->template SPtr<void>();
          kernel::wrapper::DecompressKBlockS4Fp<_GemmCore_T::PACK_ROW, _GemmCore_T::NTILE, _T>::template forward<ISA_T>(
              wptr->template WPtr<utils::int4x2>() + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2 +
                  i * KPad / 2,
              *dstptr + i * k_size, k_size, _GemmCore_T::NTILE, sptr, wptr->SDtype(), zptr, k_offset, n_offset + i,
              wptr->mBlockSize, NPad, tmpcache, cachesize);
        }

      } else if (wptr->mDType == BTLA_DTYPE::S3_CLIP) {
        auto sptr = wptr->template SPtr<void>();
        int8_t* bit3_ptr = wptr->template WPtr<int8_t>();
        auto elt_offset = n_offset * KPad + k_offset * _GemmCore_T::NTILE + i * KPad;
        assert(elt_offset % 8 == 0);
        size_t bit1_offset = size_t(NPad) * KPad;
        auto bit2ptr = reinterpret_cast<utils::bit2x4*>(bit3_ptr + elt_offset / 4);
        auto bit1ptr = reinterpret_cast<utils::bit1x8*>(bit3_ptr + bit1_offset / 4 + elt_offset / 8);
        kernel::wrapper::DecompressKBlockS3Fp<_GemmCore_T::PACK_ROW, _GemmCore_T::NTILE, _T>::template forward<ISA_T>(
            bit2ptr, bit1ptr, *dstptr + i * k_size, k_size, _GemmCore_T::NTILE, sptr, wptr->SDtype(), zptr, k_offset,
            n_offset + i, wptr->mBlockSize, NPad, tmpcache, cachesize);
      } else if (wptr->mDType == BTLA_DTYPE::S2_CLIP) {
        auto sptr = wptr->template SPtr<void>();
        int8_t* bit2_ptr = wptr->template WPtr<int8_t>();
        auto elt_offset = n_offset * KPad + k_offset * _GemmCore_T::NTILE + i * KPad;
        auto bit2ptr = reinterpret_cast<utils::bit2x4*>(bit2_ptr + elt_offset / 4);
        kernel::wrapper::DecompressKBlockS2Fp<_GemmCore_T::PACK_ROW, _GemmCore_T::NTILE, _T>::template forward<ISA_T>(
            bit2ptr, *dstptr + i * k_size, k_size, _GemmCore_T::NTILE, sptr, wptr->SDtype(), zptr, k_offset,
            n_offset + i, wptr->mBlockSize, NPad, tmpcache, cachesize);
      } else if (wptr->mDType == BTLA_DTYPE::S8) {
        auto sptr = wptr->template SPtr<void>();
        auto elt_offset = n_offset * KPad + k_offset * _GemmCore_T::NTILE + i * KPad;
        int8_t* bptr = wptr->template WPtr<int8_t>() + elt_offset;
        kernel::wrapper::DecompressKBlockS8Fp<_GemmCore_T::PACK_ROW, _GemmCore_T::NTILE, _T>::template forward<ISA_T>(
            bptr, *dstptr + i * k_size, k_size, _GemmCore_T::NTILE, sptr, wptr->SDtype(), zptr, k_offset, n_offset + i,
            wptr->mBlockSize, NPad, tmpcache, cachesize);
      } else if (wptr->mDType == BTLA_DTYPE::S5_CLIP) {
        auto sptr = wptr->template SPtr<void>();
        int8_t* bit5_ptr = wptr->template WPtr<int8_t>();
        auto elt_offset = n_offset * KPad + k_offset * _GemmCore_T::NTILE + i * KPad;
        assert(elt_offset % 8 == 0);
        size_t bit1_offset = size_t(NPad) * KPad;
        auto bit4ptr = reinterpret_cast<utils::bit4x2*>(bit5_ptr + elt_offset / 2);
        auto bit1ptr = reinterpret_cast<utils::bit1x8*>(bit5_ptr + bit1_offset / 2 + elt_offset / 8);
        kernel::wrapper::DecompressKBlockS5Fp<_GemmCore_T::PACK_ROW, _GemmCore_T::NTILE, _T>::template forward<ISA_T>(
            bit4ptr, bit1ptr, *dstptr + i * k_size, k_size, _GemmCore_T::NTILE, sptr, wptr->SDtype(), zptr, k_offset,
            n_offset + i, wptr->mBlockSize, NPad, tmpcache, cachesize);
      } else if (wptr->mDType == BTLA_DTYPE::S6_CLIP) {
        auto sptr = wptr->template SPtr<void>();
        int8_t* bit6_ptr = wptr->template WPtr<int8_t>();
        auto elt_offset = n_offset * KPad + k_offset * _GemmCore_T::NTILE + i * KPad;
        assert(elt_offset % 4 == 0);
        size_t bit2_offset = size_t(NPad) * KPad;
        auto bit4ptr = reinterpret_cast<utils::bit4x2*>(bit6_ptr + elt_offset / 2);
        auto bit2ptr = reinterpret_cast<utils::bit2x4*>(bit6_ptr + bit2_offset / 2 + elt_offset / 4);
        kernel::wrapper::DecompressKBlockS6Fp<_GemmCore_T::PACK_ROW, _GemmCore_T::NTILE, _T>::template forward<ISA_T>(
            bit4ptr, bit2ptr, *dstptr + i * k_size, k_size, _GemmCore_T::NTILE, sptr, wptr->SDtype(), zptr, k_offset,
            n_offset + i, wptr->mBlockSize, NPad, tmpcache, cachesize);
      } else if (wptr->mDType == BTLA_DTYPE::S7_CLIP) {
        auto sptr = wptr->template SPtr<void>();
        int8_t* bit7_ptr = wptr->template WPtr<int8_t>();
        auto elt_offset = n_offset * KPad + k_offset * _GemmCore_T::NTILE + i * KPad;
        assert(elt_offset % 8 == 0);
        size_t bit2_offset = size_t(NPad) * KPad;
        auto bit4ptr = reinterpret_cast<utils::bit4x2*>(bit7_ptr + elt_offset / 2);
        auto bit2ptr = reinterpret_cast<utils::bit2x4*>(bit7_ptr + bit2_offset / 2 + elt_offset / 4);
        auto bit1ptr = reinterpret_cast<utils::bit1x8*>(bit7_ptr + bit2_offset / 2 + bit2_offset / 4 + elt_offset / 8);
        kernel::wrapper::DecompressKBlockS7Fp<_GemmCore_T::PACK_ROW, _GemmCore_T::NTILE, _T>::template forward<ISA_T>(
            bit4ptr, bit2ptr, bit1ptr, *dstptr + i * k_size, k_size, _GemmCore_T::NTILE, sptr, wptr->SDtype(), zptr,
            k_offset, n_offset + i, wptr->mBlockSize, NPad, tmpcache, cachesize);
      } else if (wptr->mDType == BTLA_DTYPE::S1_CLIP) {
        auto sptr = wptr->template SPtr<void>();
        int8_t* bit1_ptr = wptr->template WPtr<int8_t>();
        auto elt_offset = n_offset * KPad + k_offset * _GemmCore_T::NTILE + i * KPad;
        assert(elt_offset % 8 == 0);
        size_t bit1_offset = size_t(NPad) * KPad;
        auto bit1ptr = reinterpret_cast<utils::bit1x8*>(bit1_ptr + elt_offset / 8);
        kernel::wrapper::DecompressKBlockS1Fp<_GemmCore_T::PACK_ROW, _GemmCore_T::NTILE, _T>::template forward<ISA_T>(
            bit1ptr, *dstptr + i * k_size, k_size, _GemmCore_T::NTILE, sptr, wptr->SDtype(), zptr, k_offset,
            n_offset + i, wptr->mBlockSize, NPad, tmpcache, cachesize);
      } else {
        assert(0);
      }
    }
    *dststep = k_size;
    return BTLA_CODE::Success;
  }

  TLACALL BTLA_CODE getQ8Weight(int8_t** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = _param.packedW;
    auto KPad = wptr->mKPad;
    auto bptr = wptr->template WPtr<int8_t>() + n_offset * KPad + k_offset * _GemmCore_T::NTILE;
    kernel::wrapper::Memcpy2D::template forward<BTLA_ISA::NoSIMD, int8_t, int8_t>(
        bptr, *dstptr, n_size / _GemmCore_T::NTILE, _GemmCore_T::NTILE * k_size, _GemmCore_T::NTILE * KPad,
        _GemmCore_T::NTILE * k_size);
    *dststep = k_size;
    return BTLA_CODE::Success;
  }

  TLACALL BTLA_CODE getQ4Weight(int8_t** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = _param.packedW;
    auto KPad = wptr->mKPad;
    auto bptr = wptr->template WPtr<utils::int4x2>() + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2;
    auto zpptr = wptr->template ZPtr<int8_t>();
    int constexpr ColSize = _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW;

    assert(wptr->mDType == BTLA_DTYPE::S4_CLIP);

    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      kernel::wrapper::DecompressKBlockS4S8<_GemmCore_T::PACK_ROW, _GemmCore_T::NTILE>::template forward<ISA_T>(
          bptr + i * KPad / 2, wptr->IsAsym() ? zpptr : nullptr, *dstptr + i * k_size, wptr->mBlockSize, wptr->CStep(),
          n_offset + i, k_offset, k_size, _GemmCore_T::NTILE, tmpcache, cachesize);
    }
    *dststep = k_size;
    return BTLA_CODE::Success;
  }

  TLACALL BTLA_CODE getQ3Weight(int8_t** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = _param.packedW;
    int8_t* bit3_ptr = wptr->template WPtr<int8_t>();
    auto zpptr = wptr->template ZPtr<int8_t>();
    auto KPad = wptr->mKPad;
    auto NPad = wptr->mNPad;
    size_t bit1_offset = size_t(NPad) * KPad;
    auto base_offset = n_offset * KPad + k_offset * _GemmCore_T::NTILE;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      auto elt_offset = base_offset + i * KPad;
      assert(elt_offset % 8 == 0);
      auto bit2ptr = reinterpret_cast<utils::bit2x4*>(bit3_ptr + elt_offset / 4);
      auto bit1ptr = reinterpret_cast<utils::bit1x8*>(bit3_ptr + bit1_offset / 4 + elt_offset / 8);
      kernel::wrapper::DecompressKBlockS3S8<_GemmCore_T::PACK_ROW, _GemmCore_T::NTILE>::template forward<ISA_T>(
          bit2ptr, bit1ptr, wptr->IsAsym() ? zpptr : nullptr, *dstptr + i * k_size, wptr->mBlockSize, wptr->CStep(),
          n_offset + i, k_offset, k_size, _GemmCore_T::NTILE, tmpcache, cachesize);
    }
    *dststep = k_size;
    return BTLA_CODE::Success;
  }

  TLACALL BTLA_CODE getQ5Weight(int8_t** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = _param.packedW;
    int8_t* bit5_ptr = wptr->template WPtr<int8_t>();
    auto zpptr = wptr->template ZPtr<int8_t>();
    auto KPad = wptr->mKPad;
    auto NPad = wptr->mNPad;
    size_t bit1_offset = size_t(NPad) * KPad;
    auto base_offset = n_offset * KPad + k_offset * _GemmCore_T::NTILE;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      auto elt_offset = base_offset + i * KPad;
      assert(elt_offset % 8 == 0);
      auto bit4ptr = reinterpret_cast<utils::bit4x2*>(bit5_ptr + elt_offset / 2);
      auto bit1ptr = reinterpret_cast<utils::bit1x8*>(bit5_ptr + bit1_offset / 2 + elt_offset / 8);
      kernel::wrapper::DecompressKBlockS5S8<_GemmCore_T::PACK_ROW, _GemmCore_T::NTILE>::template forward<ISA_T>(
          bit4ptr, bit1ptr, wptr->IsAsym() ? zpptr : nullptr, *dstptr + i * k_size, wptr->mBlockSize, wptr->CStep(),
          n_offset + i, k_offset, k_size, _GemmCore_T::NTILE, tmpcache, cachesize);
    }
    *dststep = k_size;
    return BTLA_CODE::Success;
  }

  TLACALL BTLA_CODE getQ6Weight(int8_t** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = _param.packedW;
    int8_t* bit6_ptr = wptr->template WPtr<int8_t>();
    auto zpptr = wptr->template ZPtr<int8_t>();
    auto KPad = wptr->mKPad;
    auto NPad = wptr->mNPad;
    size_t bit2_offset = size_t(NPad) * KPad;
    auto base_offset = n_offset * KPad + k_offset * _GemmCore_T::NTILE;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      auto elt_offset = base_offset + i * KPad;
      assert(elt_offset % 4 == 0);
      auto bit4ptr = reinterpret_cast<utils::bit4x2*>(bit6_ptr + elt_offset / 2);
      auto bit2ptr = reinterpret_cast<utils::bit2x4*>(bit6_ptr + bit2_offset / 2 + elt_offset / 4);
      kernel::wrapper::DecompressKBlockS6S8<_GemmCore_T::PACK_ROW, _GemmCore_T::NTILE>::template forward<ISA_T>(
          bit4ptr, bit2ptr, wptr->IsAsym() ? zpptr : nullptr, *dstptr + i * k_size, wptr->mBlockSize, wptr->CStep(),
          n_offset + i, k_offset, k_size, _GemmCore_T::NTILE, tmpcache, cachesize);
    }
    *dststep = k_size;
    return BTLA_CODE::Success;
  }

  TLACALL BTLA_CODE getQ7Weight(int8_t** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = _param.packedW;
    int8_t* bit6_ptr = wptr->template WPtr<int8_t>();
    auto zpptr = wptr->template ZPtr<int8_t>();
    auto KPad = wptr->mKPad;
    auto NPad = wptr->mNPad;
    size_t bit2_offset = size_t(NPad) * KPad;
    auto base_offset = n_offset * KPad + k_offset * _GemmCore_T::NTILE;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      auto elt_offset = base_offset + i * KPad;
      assert(elt_offset % 8 == 0);
      auto bit4ptr = reinterpret_cast<utils::bit4x2*>(bit6_ptr + elt_offset / 2);
      auto bit2ptr = reinterpret_cast<utils::bit2x4*>(bit6_ptr + bit2_offset / 2 + elt_offset / 4);
      auto bit1ptr = reinterpret_cast<utils::bit1x8*>(bit6_ptr + bit2_offset / 2 + bit2_offset / 4 + elt_offset / 8);
      kernel::wrapper::DecompressKBlockS7S8<_GemmCore_T::PACK_ROW, _GemmCore_T::NTILE>::template forward<ISA_T>(
          bit4ptr, bit2ptr, bit1ptr, wptr->IsAsym() ? zpptr : nullptr, *dstptr + i * k_size, wptr->mBlockSize,
          wptr->CStep(), n_offset + i, k_offset, k_size, _GemmCore_T::NTILE, tmpcache, cachesize);
    }
    *dststep = k_size;
    return BTLA_CODE::Success;
  }

  TLACALL BTLA_CODE getQ2Weight(int8_t** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = _param.packedW;
    int8_t* bit2_ptr = wptr->template WPtr<int8_t>();
    int8_t* zpptr = wptr->template ZPtr<int8_t>();
    auto KPad = wptr->mKPad;
    auto NPad = wptr->mNPad;
    auto base_offset = n_offset * KPad + k_offset * _GemmCore_T::NTILE;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      auto elt_offset = base_offset + i * KPad;
      assert(elt_offset % 4 == 0);
      auto bit2ptr = reinterpret_cast<utils::bit2x4*>(bit2_ptr + elt_offset / 4);
      kernel::wrapper::DecompressKBlockS2S8<_GemmCore_T::PACK_ROW, _GemmCore_T::NTILE>::template forward<ISA_T>(
          bit2ptr, wptr->IsAsym() ? zpptr : nullptr, *dstptr + i * k_size, wptr->mBlockSize, wptr->CStep(),
          n_offset + i, k_offset, k_size, _GemmCore_T::NTILE, tmpcache, cachesize);
    }
    *dststep = k_size;
    return BTLA_CODE::Success;
  }

  TLACALL BTLA_CODE getQ1Weight(int8_t** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = _param.packedW;
    int8_t* bit1_ptr = wptr->template WPtr<int8_t>();
    int8_t* zpptr = wptr->template ZPtr<int8_t>();
    auto KPad = wptr->mKPad;
    auto NPad = wptr->mNPad;
    auto base_offset = n_offset * KPad + k_offset * _GemmCore_T::NTILE;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      auto elt_offset = base_offset + i * KPad;
      assert(elt_offset % 8 == 0);
      auto bit1ptr = reinterpret_cast<utils::bit1x8*>(bit1_ptr + elt_offset / 8);
      kernel::wrapper::DecompressKBlockS1S8<_GemmCore_T::PACK_ROW, _GemmCore_T::NTILE>::template forward<ISA_T>(
          bit1ptr, wptr->IsAsym() ? zpptr : nullptr, *dstptr + i * k_size, wptr->mBlockSize, wptr->CStep(),
          n_offset + i, k_offset, k_size, _GemmCore_T::NTILE, tmpcache, cachesize);
    }
    *dststep = k_size;
    return BTLA_CODE::Success;
  }
};

struct ParamWeightKBlockNInt {
  const storage::gemm::StorageWeightNInt* stor_;
};

template <class _GemmCore_T>
class WeightKBlockNInt {
 public:
  using Storage = storage::gemm::StorageWeightNInt;
  using BType = typename _GemmCore_T::BType;
  using Param = ParamWeightKBlockNInt;

  AUTOCALL Storage create_storage(int n, int k, int blocksize, BTLA_DTYPE qtype, BTLA_DTYPE scat, bool is_asym) {
    int KPad = utils::padto(k, _GemmCore_T::KTILE);
    int NPad = utils::padto(n, _GemmCore_T::NTILE);
    Storage tmp(_GemmCore_T::ID);
    tmp.resize(NPad, KPad, blocksize <= 0 ? KPad : blocksize, n, k, qtype, storage::gemm::Layout::NPack, scat, is_asym);
    return tmp;
  }

  AUTOCALL void convert(const storage::gemm::StorageWeightNInt& src, const storage::gemm::StorageWeightNInt& dst,
                        parallel::IThreading* threading) {
    using storage::gemm::Layout;
    if (src.info_.layout_ == storage::gemm::Layout::NT) {
      convert_NT(src, dst, threading);
    }
    if (src.info_.layout_ == storage::gemm::Layout::T) {
      if (src.info_.dtype_ == BTLA_DTYPE::F32) {  // transpose to NT and quantize & reorder
        auto tmpptr = utils::amalloc<int8_t>(src.buf_size_);
        storage::gemm::StorageWeightNInt nt_stor;
        nt_stor.set_nt_data(src.info_.n_, src.info_.k_, src.info_.n_, src.info_.dtype_, tmpptr);
        transposeWeight<float>(src.info_.n_, src.info_.k_, (const float*)src.buf_.buf_ptr_, src.info_.kpad_,
                               (float*)tmpptr, src.info_.n_, threading);
        convert_NT(nt_stor, dst, threading);
        utils::afree(tmpptr);
      }
      if (src.info_.dtype_ == dst.info_.dtype_) {  // reorder
        reorder_T(src, dst, threading);
      }
    }
    if (src.info_.layout_ == storage::gemm::Layout::NPack) {
      convert_NPack(src, dst, threading);
    }
  }
  AUTOCALL void reorder_T(const storage::gemm::StorageWeightNInt& src, const storage::gemm::StorageWeightNInt& dst,
                          parallel::IThreading* threading) {
    assert(src.info_.layout_ == storage::gemm::Layout::T);
    assert(src.info_.dtype_ == dst.info_.dtype_);
    using storage::gemm::Layout;
    if (dst.info_.layout_ == storage::gemm::Layout::NPack) {
      if (src.info_.dtype_ == dst.info_.dtype_) {  // quantize and reorder
        auto tmpq = utils::amalloc<int8_t>(static_cast<size_t>(src.info_.n_) * src.info_.k_);
        utils::avector<int8_t> tmpqT(static_cast<size_t>(src.info_.n_) * src.info_.k_);
        auto ssize = static_cast<size_t>(src.info_.n_) * dst.n_block_;
        auto Tscales = utils::amalloc<float>(ssize);
        auto TscalesNT = utils::amalloc<float>(ssize);
        if (src.info_.dtype_ == BTLA_DTYPE::S8) {
          transpose(src.info_.n_, src.info_.k_, src.buf_.buf_ptr_, BTLA_DTYPE::S8, src.info_.k_, tmpq, src.info_.n_,
                    threading);
        } else {
          decompress_weight(src.info_.n_, src.info_.k_, src, tmpqT.data(), src.info_.k_, threading);
          transpose(src.info_.n_, src.info_.k_, tmpqT.data(), BTLA_DTYPE::S8, src.info_.k_, tmpq, src.info_.n_,
                    threading);
        }
        transpose(src.info_.n_, src.n_block_, src.corr_.scale_buf_.buf_ptr_, src.corr_.sdtype_, src.n_block_,
                  (int8_t*)TscalesNT, src.info_.n_, threading);
        kernel::wrapper::Memcpy2DCvtPadding::forward_auto(TscalesNT, src.corr_.sdtype_, Tscales, BTLA_DTYPE::F32,
                                                          src.n_block_, src.info_.n_, src.info_.n_, src.info_.n_, false,
                                                          tmpqT.data(), tmpqT.size());
        auto Tzps = utils::amalloc<int8_t>(dst.corr_.zp_buf_.buf_size_ ? ssize : 0);
        if (src.corr_.zptr()) {
          transpose(src.info_.n_, src.n_block_, src.corr_.zp_buf_.buf_ptr_, src.corr_.zdtype_, src.n_block_,
                    (int8_t*)Tzps, src.info_.n_, threading);
        }

        pack_qweight(src.info_.n_, src.info_.k_, tmpq, src.info_.n_, Tscales, Tzps, dst, threading);
        utils::afree(tmpq);
        utils::afree(Tscales);
        utils::afree(TscalesNT);
        utils::afree(Tzps);
      }
    }
  }

  AUTOCALL void convert_NT(const storage::gemm::StorageWeightNInt& src, const storage::gemm::StorageWeightNInt& dst,
                           parallel::IThreading* threading) {
    assert(src.info_.layout_ == storage::gemm::Layout::NT);
    assert(src.info_.dtype_ == BTLA_DTYPE::F32);
    using storage::gemm::Layout;
    if (dst.info_.layout_ == storage::gemm::Layout::NPack) {
      if (src.info_.dtype_ == BTLA_DTYPE::F32) {  // quantize and reorder
        auto tmpq = utils::amalloc<int8_t>(static_cast<size_t>(src.info_.n_) * src.info_.k_);
        auto ssize = static_cast<size_t>(src.info_.n_) * dst.n_block_;
        auto Tscales = utils::amalloc<float>(ssize);
        auto Tzps = utils::amalloc<int8_t>(dst.corr_.zp_buf_.buf_size_ ? ssize : 0);
        quantize_weight(src.info_.n_, src.info_.k_, (float*)src.buf_.buf_ptr_, src.info_.npad_, tmpq, Tscales, Tzps,
                        dst.block_, dst.info_.dtype_, threading);
        pack_qweight(src.info_.n_, src.info_.k_, tmpq, src.info_.n_, Tscales, Tzps, dst, threading);
        utils::afree(tmpq);
        utils::afree(Tscales);
        utils::afree(Tzps);
      }
    }
  }

  AUTOCALL void convert_NPack(const storage::gemm::StorageWeightNInt& src, const storage::gemm::StorageWeightNInt& dst,
                              parallel::IThreading* threading) {
    using storage::gemm::Layout;
    if (dst.info_.layout_ == Layout::NT || dst.info_.layout_ == Layout::T) {
      if (dst.info_.dtype_ == BTLA_DTYPE::F32) {
        float* upk_ptr = (float*)dst.buf_.buf_ptr_;
        if (dst.info_.layout_ == Layout::T) {
          upk_ptr = (float*)utils::amalloc<int8_t>(dst.buf_.buf_size_);
        }
        unpack_weight<float>(dst.info_.n_, dst.info_.k_, src, upk_ptr, dst.info_.n_, threading);
        if (dst.info_.layout_ == Layout::T) {
          transposeWeight<float>(dst.info_.k_, dst.info_.n_, upk_ptr, dst.info_.n_, (float*)dst.buf_.buf_ptr_,
                                 dst.info_.kpad_, threading);
          utils::afree(upk_ptr);
        }
      }
      if (dst.info_.dtype_ == src.info_.dtype_) {
        assert(dst.info_.layout_ == Layout::T);
        auto tmp_buf_NT = utils::amalloc<int8_t>(src.info_.k_ * src.info_.n_);
        if (dst.info_.dtype_ == BTLA_DTYPE::S8) {
          unpack_weight<int8_t>(dst.info_.n_, dst.info_.k_, src, tmp_buf_NT, dst.info_.n_, threading);
          transposeWeight<int8_t>(dst.info_.k_, dst.info_.n_, tmp_buf_NT, dst.info_.n_, dst.buf_.buf_ptr_, dst.info_.k_,
                                  threading);
        } else {
          auto tmp_buf_T = utils::amalloc<int8_t>(src.info_.k_ * src.info_.n_);
          unpack_weight<int8_t>(dst.info_.n_, dst.info_.k_, src, tmp_buf_NT, dst.info_.n_, threading);
          transposeWeight<int8_t>(dst.info_.k_, dst.info_.n_, tmp_buf_NT, dst.info_.n_, tmp_buf_T, dst.info_.k_,
                                  threading);
          compressWeight(dst.info_.n_, dst.info_.k_, tmp_buf_T, dst.info_.k_, dst.buf_.buf_ptr_, dst.info_.dtype_,
                         threading);
          transpose(src.n_block_, src.info_.n_, src.corr_.scale_buf_.buf_ptr_, src.corr_.sdtype_, src.info_.npad_,
                    dst.corr_.scale_buf_.buf_ptr_, dst.n_block_, threading);
          utils::afree(tmp_buf_T);
        }
        transpose(src.n_block_, src.info_.n_, src.corr_.scale_buf_.buf_ptr_, src.corr_.sdtype_, src.info_.npad_,
                  dst.corr_.scale_buf_.buf_ptr_, dst.n_block_, threading);
        if (src.corr_.zptr()) {
          assert(dst.corr_.zptr());
          transpose(src.n_block_, src.info_.n_, src.corr_.zp_buf_.buf_ptr_, src.corr_.zdtype_, src.info_.npad_,
                    dst.corr_.zp_buf_.buf_ptr_, dst.n_block_, threading);
        }
        if (src.corr_.rptr()) {
          assert(dst.corr_.rptr());
          transpose(src.n_block_, src.info_.n_, src.corr_.red_buf_.buf_ptr_, BTLA_DTYPE::S32, src.info_.npad_,
                    dst.corr_.red_buf_.buf_ptr_, dst.n_block_, threading);
        }
        utils::afree(tmp_buf_NT);
      }
    }
  }

  template <typename T>
  AUTOCALL void unpack_weight(const int N, const int K, const Storage& stor, T* B, const int ldb,
                              parallel::IThreading* threading) {
    assert(stor.info_.layout_ == storage::gemm::Layout::NPack);
    if (stor.info_.layout_ == storage::gemm::Layout::NPack) {
      parallel::Scheduler2D _para({threading->num_threads(), K, N, _GemmCore_T::KTILE, _GemmCore_T::NTILE});
      threading->parallel_for([&](int tidx) {
        parallel::ThreadProblem2D thdp{tidx};
        _para.getIndex(thdp);
        if (thdp.valid) {
          auto rowpad = utils::padto(thdp.size[0], _GemmCore_T::KTILE);
          auto colpad = utils::padto(thdp.size[1], _GemmCore_T::NTILE);
          auto dequant = utils::amalloc<T>((size_t)rowpad * colpad);
          auto dstptr = dequant;
          int dststep = 0;
          size_t constexpr CacheSize = size_t(100) << 10;
          int8_t tmpcache[CacheSize];
          GetCPUDevice();
          if (_cd->AVX512F()) {
            get_weight<BTLA_ISA::AVX512F>(&dstptr, &dststep, rowpad, colpad, thdp.loc[0], thdp.loc[1], {&stor},
                                          tmpcache, CacheSize);
          } else if (_cd->AVX2()) {
            get_weight<BTLA_ISA::AVX2>(&dstptr, &dststep, rowpad, colpad, thdp.loc[0], thdp.loc[1], {&stor}, tmpcache,
                                       CacheSize);
          } else {
            get_weight<BTLA_ISA::NoSIMD>(&dstptr, &dststep, rowpad, colpad, thdp.loc[0], thdp.loc[1], {&stor}, tmpcache,
                                         CacheSize);
          }
          kernel::wrapper::RevertPaddingInterleaveMN<_GemmCore_T::NTILE, _GemmCore_T::PACK_ROW, T>::forward_auto(
              dstptr, B + thdp.loc[0] * ldb + thdp.loc[1], thdp.size[0], thdp.size[1], rowpad, colpad, dststep, ldb);
          utils::afree(dequant);
        }
      });
    }
  }

  AUTOCALL void decompress_weight(const int N, const int K, const Storage& stor, int8_t* B, const int ldb,
                                  parallel::IThreading* threading) {
    if (stor.info_.layout_ == storage::gemm::Layout::T) {
      parallel::Scheduler2D _para({threading->num_threads(), N * K, 1, 512, 1});
      threading->parallel_for([&](int tidx) {
        parallel::ThreadProblem2D thdp{tidx};
        _para.getIndex(thdp);
        if (thdp.valid) {
          size_t constexpr CacheSize = size_t(100) << 10;
          int8_t tmpcache[CacheSize];
          GetCPUDevice();
          if (stor.info_.dtype_ == BTLA_DTYPE::S4_CLIP) {
            auto srcptr = (utils::int4x2*)stor.buf_.buf_ptr_ + thdp.loc[0] / 2;
            kernel::ref::decompress_s4_s8(srcptr, B + thdp.loc[0], thdp.size[0], tmpcache, CacheSize);
          }
        }
      });
    }
  }

  AUTOCALL void set_correction(const int N, const int K, const int8_t* zero_points, const float* scales,
                               const storage::gemm::StorageWeightNInt& stor, parallel::IThreading* threading) {
    int rawnk_scale = utils::updiv(K, stor.block_);
    int nk_scale = stor.n_block_;
    parallel::Scheduler2D _para({threading->num_threads(), 1, nk_scale, 1, 1});
    if (stor.corr_.sdtype_ == BTLA_DTYPE::BF16 || stor.corr_.sdtype_ == BTLA_DTYPE::F16 ||
        stor.corr_.sdtype_ == BTLA_DTYPE::F32) {
      threading->parallel_for([&](int tidx) {
        parallel::ThreadProblem2D thdp{tidx};
        _para.getIndex(thdp);
        if (thdp.valid) {
          int rows = thdp.loc[1] + thdp.size[1] <= rawnk_scale ? thdp.size[1] : rawnk_scale - thdp.loc[1];
          if (scales) {
            if (stor.corr_.sdtype_ == BTLA_DTYPE::BF16) {
              kernel::wrapper::Memcpy2DFp32TPadding<utils::bf16>::forward_auto(
                  scales + thdp.loc[1] * N,
                  (utils::bf16*)stor.corr_.scale_buf_.buf_ptr_ + thdp.loc[1] * stor.info_.npad_, rows, N,
                  N * sizeof(scales[0]), stor.info_.npad_ * sizeof(utils::bf16), true);
            } else if (stor.corr_.sdtype_ == BTLA_DTYPE::F32) {
              kernel::wrapper::Memcpy2DFp32TPadding<float>::forward_auto(
                  scales + thdp.loc[1] * N, (float*)stor.corr_.scale_buf_.buf_ptr_ + thdp.loc[1] * stor.info_.npad_,
                  rows, N, N * sizeof(scales[0]), stor.info_.npad_ * sizeof(float), true);
            } else if (stor.corr_.sdtype_ == BTLA_DTYPE::F16) {
              kernel::wrapper::Memcpy2DFp32TPadding<utils::fp16>::forward_auto(
                  scales + thdp.loc[1] * N,
                  (utils::fp16*)stor.corr_.scale_buf_.buf_ptr_ + thdp.loc[1] * stor.info_.npad_, rows, N,
                  N * sizeof(scales[0]), stor.info_.npad_ * sizeof(utils::fp16), true);
            }
            if (rows < thdp.size[1]) {
              auto sb = bestla::utils::bestla_dtype_bytes(stor.corr_.sdtype_);
              if (sb == 2) {
                std::memset((int16_t*)stor.corr_.scale_buf_.buf_ptr_ + (thdp.loc[1] + rows) * stor.info_.npad_, 0,
                            sb * (thdp.size[1] - rows) * stor.info_.npad_);
              } else if (sb == 4) {
                std::memset((float*)stor.corr_.scale_buf_.buf_ptr_ + (thdp.loc[1] + rows) * stor.info_.npad_, 0,
                            sb * (thdp.size[1] - rows) * stor.info_.npad_);
              } else {
                assert(0);
              }
            }
          }
          if (zero_points) {
            kernel::wrapper::Memcpy2DPadding::forward(
                zero_points + thdp.loc[1] * N, (int8_t*)stor.corr_.zp_buf_.buf_ptr_ + thdp.loc[1] * stor.info_.npad_,
                rows, N * sizeof(zero_points[0]), N * sizeof(zero_points[0]), sizeof(int8_t) * stor.info_.npad_, true);

            if (rows < thdp.size[1]) {
              std::memset((int8_t*)stor.corr_.zp_buf_.buf_ptr_ + (thdp.loc[1] + rows) * stor.info_.npad_, 0,
                          sizeof(int8_t) * (thdp.size[1] - rows) * stor.info_.npad_);
            }
          }
        }
      });
    } else {
      assert(0);
    }
  }

  AUTOCALL void pack_qweight(const int N, const int K, const int8_t* B, const int ldb, const float* scales,
                             const int8_t* zero_points, const storage::gemm::StorageWeightNInt& stor,
                             parallel::IThreading* threading) {
    set_correction(N, K, zero_points, scales, stor, threading);
    if (stor.info_.dtype_ == BTLA_DTYPE::S8) {
      reorderWeight(N, K, B, ldb, stor.buf_.buf_ptr_, threading);
    } else {
      auto reordered = utils::amalloc<int8_t>((size_t)stor.info_.kpad_ * stor.info_.npad_);
      reorderWeight(N, K, B, ldb, reordered, threading);
      compressWeight(stor.info_.npad_, stor.info_.kpad_, reordered, stor.info_.npad_, stor.buf_.buf_ptr_,
                     stor.info_.dtype_, threading);
      utils::afree(reordered);
    }
    reduce_weight(stor, threading);
  }

  AUTOCALL void reduce_weight(const Storage& stor, parallel::IThreading* threading) {
    if (stor.corr_.red_buf_.buf_size_) {
      auto deq = utils::amalloc<int8_t>((size_t)stor.info_.k_ * stor.info_.n_);
      unpack_weight(stor.info_.n_, stor.info_.k_, stor, deq, stor.info_.n_, threading);
      reduce(stor.info_.n_, stor.info_.k_, stor.block_, deq, stor.info_.n_, (int32_t*)stor.corr_.red_buf_.buf_ptr_,
             stor.corr_.step_, threading);
      utils::afree(deq);
    }
  }

  AUTOCALL void quantize_weight(const int N, const int K, const float* B, const int ldb, int8_t* qB, float* scales,
                                int8_t* zero_points, int block, BTLA_DTYPE qtype, parallel::IThreading* threading) {
    int bsize = block == -1 ? K : block;
    parallel::Scheduler2D _para({threading->num_threads(), K, N, bsize, 16});
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp({tidx});
      _para.getIndex(thdp);
      if (thdp.valid) {
        kernel::wrapper::QuantizeSignIntRowBlock::forward_auto(
            B + thdp.loc[0] * ldb + thdp.loc[1], qB + thdp.loc[0] * N + thdp.loc[1], thdp.size[0], thdp.size[1], ldb, N,
            scales + thdp.loc[0] / bsize * N + thdp.loc[1],
            zero_points == nullptr ? zero_points : zero_points + thdp.loc[0] / bsize * N + thdp.loc[1], block, qtype);
      }
    });
  }

  AUTOCALL void reorderWeight(const int N, const int K, const int8_t* B, const int ldb, int8_t* dstptr,
                              parallel::IThreading* threading) {
    int KPad = utils::padto(K, _GemmCore_T::KTILE);
    parallel::Scheduler2D _para({threading->num_threads(), K, N, _GemmCore_T::KTILE, _GemmCore_T::NTILE});
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp({tidx});
      _para.getIndex(thdp);
      if (thdp.valid) {
        auto rowpadded = utils::padto(thdp.size[0], _GemmCore_T::KTILE);
        auto colpadded = utils::padto(thdp.size[1], _GemmCore_T::NTILE);
        const auto src = B + thdp.loc[0] * ldb + thdp.loc[1];
        const auto dst = dstptr + thdp.loc[0] * _GemmCore_T::NTILE + thdp.loc[1] * KPad;
        using PaddingInterleaveMNWType =
            kernel::wrapper::PaddingInterleaveMN<_GemmCore_T::NTILE, _GemmCore_T::PACK_ROW, int8_t>;
        auto ret = PaddingInterleaveMNWType::forward_auto(  //
            src, dst, thdp.size[0], thdp.size[1], rowpadded, colpadded, ldb, KPad);
        assert(ret == BTLA_CODE::Success);
        (void)ret;
      }
    });
  }

  AUTOCALL void compressBit3Weight(const int N, const int K, const int8_t* B, int8_t* dstptr, BTLA_DTYPE qtype,
                                   parallel::IThreading* threading) {
    auto bit1_offset = size_t(N) * K;
    auto bit2ptr = reinterpret_cast<utils::bit2x4*>(dstptr);
    auto bit1ptr = reinterpret_cast<utils::bit1x8*>(dstptr + bit1_offset / 4);
    auto ret = kernel::wrapper::CompressBit3::forward_auto(B, bit2ptr, bit1ptr, bit1_offset);
    assert(ret == BTLA_CODE::Success);
  }

  AUTOCALL void compressBit5Weight(const int N, const int K, const int8_t* B, int8_t* dstptr, BTLA_DTYPE qtype,
                                   parallel::IThreading* threading) {
    auto bit1_offset = size_t(N) * K;
    auto bit4ptr = reinterpret_cast<utils::bit4x2*>(dstptr);
    auto bit1ptr = reinterpret_cast<utils::bit1x8*>(dstptr + bit1_offset / 2);
    auto ret = kernel::wrapper::CompressBit5::forward_auto(B, bit4ptr, bit1ptr, bit1_offset);
    assert(ret == BTLA_CODE::Success);
  }

  AUTOCALL void compressBit6Weight(const int N, const int K, const int8_t* B, int8_t* dstptr, BTLA_DTYPE qtype,
                                   parallel::IThreading* threading) {
    auto bit2_offset = size_t(N) * K;
    auto bit4ptr = reinterpret_cast<utils::bit4x2*>(dstptr);
    auto bit2ptr = reinterpret_cast<utils::bit2x4*>(dstptr + bit2_offset / 2);
    auto ret = kernel::wrapper::CompressBit6::forward_auto(B, bit4ptr, bit2ptr, bit2_offset);
    assert(ret == BTLA_CODE::Success);
  }

  AUTOCALL void compressBit7Weight(const int N, const int K, const int8_t* B, int8_t* dstptr, BTLA_DTYPE qtype,
                                   parallel::IThreading* threading) {
    auto eltsize = size_t(N) * K;
    auto bit4ptr = reinterpret_cast<utils::bit4x2*>(dstptr);
    auto bit2ptr = reinterpret_cast<utils::bit2x4*>(bit4ptr + eltsize / 2);
    auto bit1ptr = reinterpret_cast<utils::bit1x8*>(bit2ptr + eltsize / 4);
    auto ret = kernel::wrapper::CompressBit7::forward_auto(B, bit4ptr, bit2ptr, bit1ptr, eltsize);
    assert(ret == BTLA_CODE::Success);
  }

  AUTOCALL void compressBit2Weight(const int N, const int K, const int8_t* B, int8_t* dstptr, BTLA_DTYPE qtype,
                                   parallel::IThreading* threading) {
    // TODO(zhe): 1D parallel compress
    parallel::Scheduler2D _para({threading->num_threads(), 1, K * N, 1, 64});
    auto bit2ptr = reinterpret_cast<utils::bit2x4*>(dstptr);
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp({tidx});
      _para.getIndex(thdp);
      if (thdp.valid) {
        auto ret =
            kernel::wrapper::CompressBit2::forward_auto(B + thdp.loc[1], bit2ptr + thdp.loc[1] / 4, thdp.size[1]);
        assert(ret == BTLA_CODE::Success);
        (void)ret;
      }
    });
  }

  AUTOCALL void compressBit1Weight(const int N, const int K, const int8_t* B, int8_t* dstptr, BTLA_DTYPE qtype,
                                   parallel::IThreading* threading) {
    // TODO(zhe): 1D parallel compress
    parallel::Scheduler2D _para({threading->num_threads(), 1, K * N, 1, 64});
    auto bit1ptr = reinterpret_cast<utils::bit1x8*>(dstptr);
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp({tidx});
      _para.getIndex(thdp);
      if (thdp.valid) {
        auto ret =
            kernel::wrapper::CompressBit1::forward_auto(B + thdp.loc[1], bit1ptr + thdp.loc[1] / 8, thdp.size[1]);
        assert(ret == BTLA_CODE::Success);
        (void)ret;
      }
    });
  }

  AUTOCALL void compressBit4Weight(const int N, const int K, const int8_t* B, int8_t* dstptr, BTLA_DTYPE qtype,
                                   parallel::IThreading* threading) {
    parallel::Scheduler2D _para({threading->num_threads(), 1, K * N, 1, 64});
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp({tidx});
      _para.getIndex(thdp);
      if (thdp.valid) {
        BTLA_CODE ret = BTLA_CODE::NotSupport;
        if (qtype == BTLA_DTYPE::S4_CLIP) {
          auto bit4ptr = reinterpret_cast<utils::int4x2*>(dstptr);
          ret = kernel::wrapper::CompressS8S4::forward_auto(B + thdp.loc[1], bit4ptr + thdp.loc[1] / 2, thdp.size[1]);
        } else if (qtype == BTLA_DTYPE::F4_BNB || qtype == BTLA_DTYPE::F4_NF4 || qtype == BTLA_DTYPE::F4_E2M1) {
          auto bit4ptr = reinterpret_cast<utils::f4x2*>(dstptr);
          ret = kernel::wrapper::CompressFp4::forward_auto(B + thdp.loc[1], bit4ptr + thdp.loc[1] / 2, thdp.size[1]);
        } else {
          assert(0);
        }
        assert(ret == BTLA_CODE::Success);
        (void)ret;
      }
    });
  }

  AUTOCALL void compressWeight(const int N, const int K, const int8_t* B, const int ldb, int8_t* dstptr,
                               BTLA_DTYPE qtype, parallel::IThreading* threading) {
    if (qtype == BTLA_DTYPE::S7_CLIP) return compressBit7Weight(N, K, B, dstptr, qtype, threading);
    if (qtype == BTLA_DTYPE::S6_CLIP) return compressBit6Weight(N, K, B, dstptr, qtype, threading);
    if (qtype == BTLA_DTYPE::S5_CLIP) return compressBit5Weight(N, K, B, dstptr, qtype, threading);
    if (qtype == BTLA_DTYPE::S4_CLIP) return compressBit4Weight(N, K, B, dstptr, qtype, threading);
    if (qtype == BTLA_DTYPE::S3_CLIP) return compressBit3Weight(N, K, B, dstptr, qtype, threading);
    if (qtype == BTLA_DTYPE::S2_CLIP) return compressBit2Weight(N, K, B, dstptr, qtype, threading);
    if (qtype == BTLA_DTYPE::S1_CLIP) return compressBit1Weight(N, K, B, dstptr, qtype, threading);
    if (qtype == BTLA_DTYPE::F4_BNB || qtype == BTLA_DTYPE::F4_NF4 || qtype == BTLA_DTYPE::F4_E2M1)
      return compressBit4Weight(N, K, B, dstptr, qtype, threading);
  }

  AUTOCALL void reduce(const int N, const int K, const int KBlock, const int8_t* B, const int ldb, int32_t* rptr,
                       const int ldr, parallel::IThreading* threading) {
    parallel::Scheduler2D _para({threading->num_threads(), K, N, KBlock, 16});
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp({tidx});
      _para.getIndex(thdp);
      if (thdp.valid) {
        const auto src = B + thdp.loc[0] * ldb + thdp.loc[1];
        const auto dst = rptr + thdp.loc[1] + thdp.loc[0] / KBlock * ldr;
        using RowReduceSum = kernel::wrapper::RowReduceSum<int8_t, int32_t>;
        for (int i = 0; i < thdp.size[0]; i += KBlock) {
          int rowremain = utils::remainsize(thdp.loc[0] + i, K, KBlock);
          auto ret = RowReduceSum::forward_auto(src + i * ldb, ldb, rowremain, thdp.size[1], dst + i / KBlock * ldr);
          assert(ret == BTLA_CODE::Success);
          (void)ret;
        }
      }
    });
  }

 public:
  template <BTLA_ISA ISA_T, typename _T>
  static inline BTLA_CODE get_weight(_T** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                     const Param& _param, void* tmpcache, size_t cachesize) {
    if constexpr (std::is_same_v<_T, int8_t>) {
      return get_int<ISA_T>(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
    }
    return get_fp<ISA_T>(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
  }

  TLACALL BTLA_CODE get_int(int8_t** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                            const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = _param.stor_;
    if (wptr->info_.dtype_ == BTLA_DTYPE::S8) {
      return getQ8Weight<ISA_T>(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
    } else if (wptr->info_.dtype_ == BTLA_DTYPE::S6_CLIP) {
      return getQ6Weight<ISA_T>(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
    } else if (wptr->info_.dtype_ == BTLA_DTYPE::S5_CLIP) {
      return getQ5Weight<ISA_T>(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
    } else if (wptr->info_.dtype_ == BTLA_DTYPE::S4_CLIP) {
      return getQ4Weight<ISA_T>(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
    } else if (wptr->info_.dtype_ == BTLA_DTYPE::S3_CLIP) {
      return getQ3Weight<ISA_T>(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
    } else if (wptr->info_.dtype_ == BTLA_DTYPE::S2_CLIP) {
      return getQ2Weight<ISA_T>(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
    } else if (wptr->info_.dtype_ == BTLA_DTYPE::S7_CLIP) {
      return getQ7Weight<ISA_T>(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
    } else if (wptr->info_.dtype_ == BTLA_DTYPE::S1_CLIP) {
      return getQ1Weight<ISA_T>(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
    } else {
      assert(0);
    }
    return BTLA_CODE::NotSupport;
  }

  TLACALL BTLA_CODE get_scale(float** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                              const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = _param.stor_;
    if (wptr->corr_.sdtype_ == BTLA_DTYPE::F32) {
      auto aptr = (float*)wptr->corr_.scale_buf_.buf_ptr_;
      kernel::wrapper::Memcpy2D::template forward<BTLA_ISA::NoSIMD>(
          aptr + k_offset / wptr->block_ * wptr->corr_.step_ + n_offset, *dstptr, utils::updiv(k_size, wptr->block_),
          n_size, wptr->corr_.step_, n_size);
      *dststep = n_size;
    }
    if (wptr->corr_.sdtype_ == BTLA_DTYPE::BF16) {
      auto aptr = (utils::bf16*)wptr->corr_.scale_buf_.buf_ptr_;
      kernel::wrapper::Memcpy2DBf16CvtFp32::forward<ISA_T>(
          aptr + k_offset / wptr->block_ * wptr->corr_.step_ + n_offset, *dstptr, utils::updiv(k_size, wptr->block_),
          n_size, wptr->corr_.step_ * 2, n_size * 4, false);
      *dststep = n_size;
    }
    if (wptr->corr_.sdtype_ == BTLA_DTYPE::F16) {
      auto aptr = (utils::fp16*)wptr->corr_.scale_buf_.buf_ptr_;
      kernel::wrapper::Memcpy2DFp16CvtFp32::forward<ISA_T>(
          aptr + k_offset / wptr->block_ * wptr->corr_.step_ + n_offset, *dstptr, utils::updiv(k_size, wptr->block_),
          n_size, wptr->corr_.step_ * 2, n_size * 4, false);
      *dststep = n_size;
    }
    /*if (wptr->corr_.sdtype_ == BTLA_DTYPE::DQ8_BNB) {
      auto aptr = wptr->template SPtr<uint8_t>();
      auto internal_k_offset = k_offset / wptr->block_;
      auto dq_offset_idx = static_cast<int>(wptr->mCorrection.mDQCorrectionBuf.mBufSize / sizeof(float) - 1);
      kernel::wrapper::Dq8GetScale::template forward<ISA_T>(
          aptr + internal_k_offset * wptr->corr_.step_ + n_offset, *dstptr, utils::updiv(k_size, wptr->block_), n_size,
          internal_k_offset * wptr->mN + n_offset, wptr->mDqBlockSize, dq_offset_idx, wptr->DQPtr<float>(),
          wptr->corr_.step_, n_size, false, wptr->mN);
    }*/
    return BTLA_CODE::Success;
  }

  TLACALL BTLA_CODE get_reduce(int32_t** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                               const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = _param.stor_;
    auto rptr = (int32_t*)wptr->corr_.red_buf_.buf_ptr_;
    kernel::wrapper::Memcpy2D::template forward<BTLA_ISA::NoSIMD>(
        rptr + k_offset / wptr->block_ * wptr->corr_.step_ + n_offset, *dstptr, utils::updiv(k_size, wptr->block_),
        n_size, wptr->corr_.step_, n_size);
    *dststep = n_size;
    return BTLA_CODE::Success;
  }

 protected:
  template <BTLA_ISA ISA_T, typename _T>
  static inline BTLA_CODE get_fp(_T** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                 const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = _param.stor_;
    auto NPad = wptr->info_.npad_;
    auto KPad = wptr->info_.kpad_;
    int constexpr ColSize = _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      auto zptr = wptr->corr_.zptr();
      auto sptr = wptr->corr_.scale_buf_.buf_ptr_;
      if (wptr->info_.dtype_ == BTLA_DTYPE::S4_CLIP) {
        kernel::wrapper::DecompressKBlockS4Fp<_GemmCore_T::PACK_ROW, _GemmCore_T::NTILE, _T>::template forward<ISA_T>(
            (utils::int4x2*)wptr->buf_.buf_ptr_ + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2 +
                i * KPad / 2,
            *dstptr + i * k_size, k_size, _GemmCore_T::NTILE, sptr, wptr->corr_.sdtype_, zptr, k_offset, n_offset + i,
            wptr->block_, NPad, tmpcache, cachesize);

      } else if (wptr->info_.dtype_ == BTLA_DTYPE::S3_CLIP) {
        int8_t* bit3_ptr = wptr->buf_.buf_ptr_;
        auto elt_offset = n_offset * KPad + k_offset * _GemmCore_T::NTILE + i * KPad;
        assert(elt_offset % 8 == 0);
        size_t bit1_offset = size_t(NPad) * KPad;
        auto bit2ptr = reinterpret_cast<utils::bit2x4*>(bit3_ptr + elt_offset / 4);
        auto bit1ptr = reinterpret_cast<utils::bit1x8*>(bit3_ptr + bit1_offset / 4 + elt_offset / 8);
        kernel::wrapper::DecompressKBlockS3Fp<_GemmCore_T::PACK_ROW, _GemmCore_T::NTILE, _T>::template forward<ISA_T>(
            bit2ptr, bit1ptr, *dstptr + i * k_size, k_size, _GemmCore_T::NTILE, sptr, wptr->corr_.sdtype_, zptr,
            k_offset, n_offset + i, wptr->block_, NPad, tmpcache, cachesize);
      } else if (wptr->info_.dtype_ == BTLA_DTYPE::S2_CLIP) {
        int8_t* bit2_ptr = wptr->buf_.buf_ptr_;
        auto elt_offset = n_offset * KPad + k_offset * _GemmCore_T::NTILE + i * KPad;
        auto bit2ptr = reinterpret_cast<utils::bit2x4*>(bit2_ptr + elt_offset / 4);
        kernel::wrapper::DecompressKBlockS2Fp<_GemmCore_T::PACK_ROW, _GemmCore_T::NTILE, _T>::template forward<ISA_T>(
            bit2ptr, *dstptr + i * k_size, k_size, _GemmCore_T::NTILE, sptr, wptr->corr_.sdtype_, zptr, k_offset,
            n_offset + i, wptr->block_, NPad, tmpcache, cachesize);
      } else if (wptr->info_.dtype_ == BTLA_DTYPE::S8) {
        auto elt_offset = n_offset * KPad + k_offset * _GemmCore_T::NTILE + i * KPad;
        int8_t* bptr = wptr->buf_.buf_ptr_ + elt_offset;
        kernel::wrapper::DecompressKBlockS8Fp<_GemmCore_T::PACK_ROW, _GemmCore_T::NTILE, _T>::template forward<ISA_T>(
            bptr, *dstptr + i * k_size, k_size, _GemmCore_T::NTILE, sptr, wptr->corr_.sdtype_, zptr, k_offset,
            n_offset + i, wptr->block_, NPad, tmpcache, cachesize);
      } else if (wptr->info_.dtype_ == BTLA_DTYPE::S5_CLIP) {
        int8_t* bit5_ptr = wptr->buf_.buf_ptr_;
        auto elt_offset = n_offset * KPad + k_offset * _GemmCore_T::NTILE + i * KPad;
        assert(elt_offset % 8 == 0);
        size_t bit1_offset = size_t(NPad) * KPad;
        auto bit4ptr = reinterpret_cast<utils::bit4x2*>(bit5_ptr + elt_offset / 2);
        auto bit1ptr = reinterpret_cast<utils::bit1x8*>(bit5_ptr + bit1_offset / 2 + elt_offset / 8);
        kernel::wrapper::DecompressKBlockS5Fp<_GemmCore_T::PACK_ROW, _GemmCore_T::NTILE, _T>::template forward<ISA_T>(
            bit4ptr, bit1ptr, *dstptr + i * k_size, k_size, _GemmCore_T::NTILE, sptr, wptr->corr_.sdtype_, zptr,
            k_offset, n_offset + i, wptr->block_, NPad, tmpcache, cachesize);
      } else if (wptr->info_.dtype_ == BTLA_DTYPE::S6_CLIP) {
        int8_t* bit6_ptr = wptr->buf_.buf_ptr_;
        auto elt_offset = n_offset * KPad + k_offset * _GemmCore_T::NTILE + i * KPad;
        assert(elt_offset % 4 == 0);
        size_t bit2_offset = size_t(NPad) * KPad;
        auto bit4ptr = reinterpret_cast<utils::bit4x2*>(bit6_ptr + elt_offset / 2);
        auto bit2ptr = reinterpret_cast<utils::bit2x4*>(bit6_ptr + bit2_offset / 2 + elt_offset / 4);
        kernel::wrapper::DecompressKBlockS6Fp<_GemmCore_T::PACK_ROW, _GemmCore_T::NTILE, _T>::template forward<ISA_T>(
            bit4ptr, bit2ptr, *dstptr + i * k_size, k_size, _GemmCore_T::NTILE, sptr, wptr->corr_.sdtype_, zptr,
            k_offset, n_offset + i, wptr->block_, NPad, tmpcache, cachesize);
      } else if (wptr->info_.dtype_ == BTLA_DTYPE::S7_CLIP) {
        int8_t* bit7_ptr = wptr->buf_.buf_ptr_;
        auto elt_offset = n_offset * KPad + k_offset * _GemmCore_T::NTILE + i * KPad;
        assert(elt_offset % 8 == 0);
        size_t bit2_offset = size_t(NPad) * KPad;
        auto bit4ptr = reinterpret_cast<utils::bit4x2*>(bit7_ptr + elt_offset / 2);
        auto bit2ptr = reinterpret_cast<utils::bit2x4*>(bit7_ptr + bit2_offset / 2 + elt_offset / 4);
        auto bit1ptr = reinterpret_cast<utils::bit1x8*>(bit7_ptr + bit2_offset / 2 + bit2_offset / 4 + elt_offset / 8);
        kernel::wrapper::DecompressKBlockS7Fp<_GemmCore_T::PACK_ROW, _GemmCore_T::NTILE, _T>::template forward<ISA_T>(
            bit4ptr, bit2ptr, bit1ptr, *dstptr + i * k_size, k_size, _GemmCore_T::NTILE, sptr, wptr->corr_.sdtype_,
            zptr, k_offset, n_offset + i, wptr->block_, NPad, tmpcache, cachesize);
      } else if (wptr->info_.dtype_ == BTLA_DTYPE::S1_CLIP) {
        int8_t* bit1_ptr = wptr->buf_.buf_ptr_;
        auto elt_offset = n_offset * KPad + k_offset * _GemmCore_T::NTILE + i * KPad;
        assert(elt_offset % 8 == 0);
        size_t bit1_offset = size_t(NPad) * KPad;
        auto bit1ptr = reinterpret_cast<utils::bit1x8*>(bit1_ptr + elt_offset / 8);
        kernel::wrapper::DecompressKBlockS1Fp<_GemmCore_T::PACK_ROW, _GemmCore_T::NTILE, _T>::template forward<ISA_T>(
            bit1ptr, *dstptr + i * k_size, k_size, _GemmCore_T::NTILE, sptr, wptr->corr_.sdtype_, zptr, k_offset,
            n_offset + i, wptr->block_, NPad, tmpcache, cachesize);
      } else {
        assert(0);
      }
    }
    *dststep = k_size;
    return BTLA_CODE::Success;
  }

  TLACALL BTLA_CODE getQ8Weight(int8_t** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = _param.stor_;
    auto KPad = wptr->info_.kpad_;
    auto bptr = wptr->buf_.buf_ptr_ + n_offset * KPad + k_offset * _GemmCore_T::NTILE;
    kernel::wrapper::Memcpy2D::template forward<BTLA_ISA::NoSIMD, int8_t, int8_t>(
        bptr, *dstptr, n_size / _GemmCore_T::NTILE, _GemmCore_T::NTILE * k_size, _GemmCore_T::NTILE * KPad,
        _GemmCore_T::NTILE * k_size);
    *dststep = k_size;
    return BTLA_CODE::Success;
  }

  TLACALL BTLA_CODE getQ4Weight(int8_t** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = _param.stor_;
    auto KPad = wptr->info_.kpad_;
    using storage::gemm::Layout;
    if (wptr->info_.layout_ == Layout::NPack) {
      auto bptr = (utils::int4x2*)wptr->buf_.buf_ptr_ + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2;
      auto zpptr = wptr->corr_.zptr();
      int constexpr ColSize = _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW;
      for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
        kernel::wrapper::DecompressKBlockS4S8<_GemmCore_T::PACK_ROW, _GemmCore_T::NTILE>::template forward<ISA_T>(
            bptr + i * KPad / 2, zpptr, *dstptr + i * k_size, wptr->block_, wptr->corr_.step_, n_offset + i, k_offset,
            k_size, _GemmCore_T::NTILE, tmpcache, cachesize);
      }
      *dststep = k_size;
    }
    return BTLA_CODE::Success;
  }

  TLACALL BTLA_CODE getQ3Weight(int8_t** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = _param.stor_;
    int8_t* bit3_ptr = wptr->buf_.buf_ptr_;
    auto zpptr = wptr->corr_.zptr();
    auto KPad = wptr->info_.kpad_;
    auto NPad = wptr->info_.npad_;
    size_t bit1_offset = size_t(NPad) * KPad;
    auto base_offset = n_offset * KPad + k_offset * _GemmCore_T::NTILE;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      auto elt_offset = base_offset + i * KPad;
      assert(elt_offset % 8 == 0);
      auto bit2ptr = reinterpret_cast<utils::bit2x4*>(bit3_ptr + elt_offset / 4);
      auto bit1ptr = reinterpret_cast<utils::bit1x8*>(bit3_ptr + bit1_offset / 4 + elt_offset / 8);
      kernel::wrapper::DecompressKBlockS3S8<_GemmCore_T::PACK_ROW, _GemmCore_T::NTILE>::template forward<ISA_T>(
          bit2ptr, bit1ptr, zpptr, *dstptr + i * k_size, wptr->block_, wptr->corr_.step_, n_offset + i, k_offset,
          k_size, _GemmCore_T::NTILE, tmpcache, cachesize);
    }
    *dststep = k_size;
    return BTLA_CODE::Success;
  }

  TLACALL BTLA_CODE getQ5Weight(int8_t** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = _param.stor_;
    int8_t* bit5_ptr = wptr->buf_.buf_ptr_;
    auto zpptr = wptr->corr_.zptr();
    auto KPad = wptr->info_.kpad_;
    auto NPad = wptr->info_.npad_;
    size_t bit1_offset = size_t(NPad) * KPad;
    auto base_offset = n_offset * KPad + k_offset * _GemmCore_T::NTILE;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      auto elt_offset = base_offset + i * KPad;
      assert(elt_offset % 8 == 0);
      auto bit4ptr = reinterpret_cast<utils::bit4x2*>(bit5_ptr + elt_offset / 2);
      auto bit1ptr = reinterpret_cast<utils::bit1x8*>(bit5_ptr + bit1_offset / 2 + elt_offset / 8);
      kernel::wrapper::DecompressKBlockS5S8<_GemmCore_T::PACK_ROW, _GemmCore_T::NTILE>::template forward<ISA_T>(
          bit4ptr, bit1ptr, zpptr, *dstptr + i * k_size, wptr->block_, wptr->corr_.step_, n_offset + i, k_offset,
          k_size, _GemmCore_T::NTILE, tmpcache, cachesize);
    }
    *dststep = k_size;
    return BTLA_CODE::Success;
  }

  TLACALL BTLA_CODE getQ6Weight(int8_t** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = _param.stor_;
    int8_t* bit6_ptr = wptr->buf_.buf_ptr_;
    auto zpptr = wptr->corr_.zptr();
    auto KPad = wptr->info_.kpad_;
    auto NPad = wptr->info_.npad_;
    size_t bit2_offset = size_t(NPad) * KPad;
    auto base_offset = n_offset * KPad + k_offset * _GemmCore_T::NTILE;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      auto elt_offset = base_offset + i * KPad;
      assert(elt_offset % 4 == 0);
      auto bit4ptr = reinterpret_cast<utils::bit4x2*>(bit6_ptr + elt_offset / 2);
      auto bit2ptr = reinterpret_cast<utils::bit2x4*>(bit6_ptr + bit2_offset / 2 + elt_offset / 4);
      kernel::wrapper::DecompressKBlockS6S8<_GemmCore_T::PACK_ROW, _GemmCore_T::NTILE>::template forward<ISA_T>(
          bit4ptr, bit2ptr, zpptr, *dstptr + i * k_size, wptr->block_, wptr->corr_.step_, n_offset + i, k_offset,
          k_size, _GemmCore_T::NTILE, tmpcache, cachesize);
    }
    *dststep = k_size;
    return BTLA_CODE::Success;
  }

  TLACALL BTLA_CODE getQ7Weight(int8_t** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = _param.stor_;
    int8_t* bit6_ptr = wptr->buf_.buf_ptr_;
    auto zpptr = wptr->corr_.zptr();
    auto KPad = wptr->info_.kpad_;
    auto NPad = wptr->info_.npad_;
    size_t bit2_offset = size_t(NPad) * KPad;
    auto base_offset = n_offset * KPad + k_offset * _GemmCore_T::NTILE;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      auto elt_offset = base_offset + i * KPad;
      assert(elt_offset % 8 == 0);
      auto bit4ptr = reinterpret_cast<utils::bit4x2*>(bit6_ptr + elt_offset / 2);
      auto bit2ptr = reinterpret_cast<utils::bit2x4*>(bit6_ptr + bit2_offset / 2 + elt_offset / 4);
      auto bit1ptr = reinterpret_cast<utils::bit1x8*>(bit6_ptr + bit2_offset / 2 + bit2_offset / 4 + elt_offset / 8);
      kernel::wrapper::DecompressKBlockS7S8<_GemmCore_T::PACK_ROW, _GemmCore_T::NTILE>::template forward<ISA_T>(
          bit4ptr, bit2ptr, bit1ptr, zpptr, *dstptr + i * k_size, wptr->block_, wptr->corr_.step_, n_offset + i,
          k_offset, k_size, _GemmCore_T::NTILE, tmpcache, cachesize);
    }
    *dststep = k_size;
    return BTLA_CODE::Success;
  }

  TLACALL BTLA_CODE getQ2Weight(int8_t** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = _param.stor_;
    int8_t* bit2_ptr = wptr->buf_.buf_ptr_;
    int8_t* zpptr = wptr->corr_.zptr();
    auto KPad = wptr->info_.kpad_;
    auto NPad = wptr->info_.npad_;
    auto base_offset = n_offset * KPad + k_offset * _GemmCore_T::NTILE;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      auto elt_offset = base_offset + i * KPad;
      assert(elt_offset % 4 == 0);
      auto bit2ptr = reinterpret_cast<utils::bit2x4*>(bit2_ptr + elt_offset / 4);
      kernel::wrapper::DecompressKBlockS2S8<_GemmCore_T::PACK_ROW, _GemmCore_T::NTILE>::template forward<ISA_T>(
          bit2ptr, zpptr, *dstptr + i * k_size, wptr->block_, wptr->corr_.step_, n_offset + i, k_offset, k_size,
          _GemmCore_T::NTILE, tmpcache, cachesize);
    }
    *dststep = k_size;
    return BTLA_CODE::Success;
  }

  TLACALL BTLA_CODE getQ1Weight(int8_t** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = _param.stor_;
    int8_t* bit1_ptr = wptr->buf_.buf_ptr_;
    int8_t* zpptr = wptr->corr_.zptr();
    auto KPad = wptr->info_.kpad_;
    auto NPad = wptr->info_.npad_;
    auto base_offset = n_offset * KPad + k_offset * _GemmCore_T::NTILE;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      auto elt_offset = base_offset + i * KPad;
      assert(elt_offset % 8 == 0);
      auto bit1ptr = reinterpret_cast<utils::bit1x8*>(bit1_ptr + elt_offset / 8);
      kernel::wrapper::DecompressKBlockS1S8<_GemmCore_T::PACK_ROW, _GemmCore_T::NTILE>::template forward<ISA_T>(
          bit1ptr, zpptr, *dstptr + i * k_size, wptr->block_, wptr->corr_.step_, n_offset + i, k_offset, k_size,
          _GemmCore_T::NTILE, tmpcache, cachesize);
    }
    *dststep = k_size;
    return BTLA_CODE::Success;
  }
};

struct ParamWeightKBlockNFloat {
  storage::gemm::StorageWeightKBlockNFloat* packedW;
};

template <class _GemmCore_T>
class WeightKBlockNFloat {
 public:
  using Param = ParamWeightKBlockNFloat;  // NFloat storage Param same with NInteger storage.
  using StorageWeight = storage::gemm::StorageWeightKBlockNFloat;
  using QuantBaseT = WeightKBlockNInteger<_GemmCore_T>;
  AUTOCALL StorageWeight createStorage(const int N, const int K, int blocksize, BTLA_DTYPE fT, BTLA_DTYPE scaT) {
    int KPad = utils::padto(K, _GemmCore_T::KTILE);
    int NPad = utils::padto(N, _GemmCore_T::NTILE);
    StorageWeight tmp(_GemmCore_T::ID);
    tmp.resize(NPad, KPad, blocksize <= 0 ? KPad : blocksize, N, K, fT, scaT);
    return tmp;
  }

  AUTOCALL void packTransposeWeight(const int N, const int K, const float* B, const int ldb, StorageWeight* stor,
                                    parallel::IThreading* threading) {
    auto B_NT = utils::amalloc<float>(static_cast<size_t>(N) * K);
    transposeWeight<float>(N, K, B, ldb, B_NT, N, threading);
    packWeight(N, K, B_NT, N, stor, threading);
    utils::afree(B_NT);
  }

  // from KxN f32 weight to packed N//NtilexKPadxNTile int8 weight
  AUTOCALL void packWeight(const int N, const int K, const float* B, const int ldb, StorageWeight* ptr,
                           parallel::IThreading* threading) {
    auto tmpq = utils::amalloc<int8_t>(static_cast<size_t>(N) * K);
    int nk_scale = utils::updiv(K, ptr->mBlockSize);
    auto ssize = static_cast<size_t>(N) * nk_scale;
    auto Tscales = utils::amalloc<float>(ssize);
    auto Tzps = utils::amalloc<int8_t>(ptr->IsAsym() ? ssize : 0);
    quantizeWeight(N, K, B, ldb, tmpq, Tscales, Tzps, ptr, threading);
    packQWeight(N, K, tmpq, N, Tscales, Tzps, ptr, threading);
    utils::afree(tmpq);
    utils::afree(Tscales);
    utils::afree(Tzps);
  }

  AUTOCALL void quantizeWeight(const int N, const int K, const float* B, const int ldb, int8_t* qB, float* scales,
                               int8_t* zero_points, void* stor, parallel::IThreading* threading) {
    auto ptr = reinterpret_cast<StorageWeight*>(stor);
    int bsize = ptr->mBlockSize == -1 ? K : ptr->mBlockSize;
    parallel::Scheduler2D _para({threading->num_threads(), K, N, bsize, 16});
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp({tidx});
      _para.getIndex(thdp);
      if (thdp.valid) {
        quantRowBlock(B + thdp.loc[0] * ldb + thdp.loc[1], qB + thdp.loc[0] * N + thdp.loc[1], thdp.size[0],
                      thdp.size[1], ldb, N, scales + thdp.loc[0] / bsize * N + thdp.loc[1], ptr);
      }
    });
  }

  // from packed N//NtilexKPadxNTile int8 weight to KxN f32 weight
  AUTOCALL void unpackTransposeWeight(const int N, const int K, StorageWeight* stor, float* B, const int ldb,
                                      parallel::IThreading* threading) {
    auto B_NT = utils::amalloc<float>(static_cast<size_t>(N) * K);
    unpackWeight(N, K, stor, B_NT, N, threading);
    transposeWeight<float>(K, N, B_NT, N, B, ldb, threading);
    utils::afree(B_NT);
  }

  AUTOCALL void unpackWeight(const int N, const int K, StorageWeight* stor, float* B, const int ldb,
                             parallel::IThreading* threading) {
    parallel::Scheduler2D _para({threading->num_threads(), K, N, _GemmCore_T::KTILE, _GemmCore_T::NTILE});
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp{tidx};
      _para.getIndex(thdp);
      if (thdp.valid) {
        auto rowpad = utils::padto(thdp.size[0], _GemmCore_T::KTILE);
        auto colpad = utils::padto(thdp.size[1], _GemmCore_T::NTILE);
        auto dequant = utils::amalloc<float>((size_t)rowpad * colpad);
        auto dstptr = dequant;
        int dststep = 0;
        size_t constexpr CacheSize = size_t(100) << 10;
        int8_t tmpcache[CacheSize];
        GetCPUDevice();
        if (_cd->AVX512F()) {
          getWeight<BTLA_ISA::AVX512F>(&dstptr, &dststep, rowpad, colpad, thdp.loc[0], thdp.loc[1], {stor}, tmpcache,
                                       CacheSize);
        } else if (_cd->AVX2()) {
          getWeight<BTLA_ISA::AVX2>(&dstptr, &dststep, rowpad, colpad, thdp.loc[0], thdp.loc[1], {stor}, tmpcache,
                                    CacheSize);
        } else {
          getWeight<BTLA_ISA::NoSIMD>(&dstptr, &dststep, rowpad, colpad, thdp.loc[0], thdp.loc[1], {stor}, tmpcache,
                                      CacheSize);
        }
        kernel::wrapper::RevertPaddingInterleaveMN<_GemmCore_T::NTILE, _GemmCore_T::PACK_ROW, float>::forward_auto(
            dstptr, B + thdp.loc[0] * ldb + thdp.loc[1], thdp.size[0], thdp.size[1], rowpad, colpad, dststep, ldb);
        utils::afree(dequant);
      }
    });
  }

  AUTOCALL void setDoubleQuantCorrection(utils::avector<float>* dq_buf, StorageWeight* ptr) {
    if (ptr->SDtype() == BTLA_DTYPE::DQ8_BNB) {
      auto packw_dqbuf_ptr = ptr->DQPtr<float>();
      memcpy(packw_dqbuf_ptr, dq_buf->data(), dq_buf->size() * sizeof(float));
    } else {
      assert(0);
    }
  }

  AUTOCALL void packQWeight(const int N, const int K, const int8_t* B, const int ldb, const float* scales,
                            const int8_t* zero_points, StorageWeight* stor, parallel::IThreading* threading) {
    if (stor->IsDoubleQuant()) {
      int nk_scale = utils::updiv(K, stor->mBlockSize);
      auto ssize = static_cast<size_t>(N) * nk_scale;
      utils::avector<float> dq_buf;
      QuantBaseT::doubleQuantScale(const_cast<float*>(scales), ssize, stor->mDqBlockSize, stor->SDtype(), &dq_buf);
      setDoubleQuantCorrection(&dq_buf, stor);
    }
    setQuantCorrection(N, K, zero_points, scales, stor, threading);
    if (stor->mDType == BTLA_DTYPE::F8_E4M3 || stor->mDType == BTLA_DTYPE::F8_E5M2) {
      QuantBaseT::reorderWeight(N, K, B, ldb, stor->WPtr<int8_t>(), threading);
    } else {
      auto reordered = utils::amalloc<int8_t>((size_t)stor->mKPad * stor->mNPad);
      QuantBaseT::reorderWeight(N, K, B, ldb, reordered, threading);
      QuantBaseT::compressWeight(stor->mNPad, stor->mKPad, reordered, stor->mNPad, stor->WPtr<int8_t>(), stor->mDType,
                                 threading);
      utils::afree(reordered);
    }
  }

  AUTOCALL void setQuantCorrection(const int N, const int K, const int8_t* zero_points, const float* scales,
                                   StorageWeight* stor, parallel::IThreading* threading) {
    int rawnk_scale = utils::updiv(K, stor->mBlockSize);
    int nk_scale = utils::updiv(stor->mKPad, stor->mBlockSize);
    parallel::Scheduler2D _para({threading->num_threads(), 1, nk_scale, 1, 1});
    if (stor->SDtype() == BTLA_DTYPE::F32) {  // fp32 to fp32 direct copy
      threading->parallel_for([&](int tidx) {
        parallel::ThreadProblem2D thdp{tidx};
        _para.getIndex(thdp);
        if (thdp.valid) {
          for (int i = thdp.loc[1]; i < thdp.loc[1] + thdp.size[1]; i++) {
            if (i < rawnk_scale) {
              if (scales != nullptr)
                std::memcpy(stor->template SPtr<float>() + i * stor->mNPad, scales + i * N, N * sizeof(scales[0]));
              if (zero_points != nullptr)
                std::memcpy(stor->template ZPtr<int8_t>() + i * stor->mNPad, zero_points + i * N,
                            N * sizeof(zero_points[0]));
            } else {
              if (scales != nullptr)
                std::memset(stor->template SPtr<float>() + i * stor->mNPad, 0, stor->mNPad * sizeof(float));
              if (zero_points != nullptr)
                std::memset(stor->template ZPtr<int8_t>() + i * stor->mNPad, 0, stor->mNPad * sizeof(zero_points[0]));
            }
          }
        }
      });
    } else if (stor->SDtype() == BTLA_DTYPE::BF16) {
      threading->parallel_for([&](int tidx) {
        parallel::ThreadProblem2D thdp{tidx};
        _para.getIndex(thdp);
        if (thdp.valid) {
          for (int i = thdp.loc[1]; i < thdp.loc[1] + thdp.size[1]; i++) {
            if (i < rawnk_scale) {
              if (scales != nullptr) {
                for (int j = 0; j < N; j++) {
                  stor->template SPtr<utils::bf16>()[j + i * stor->mNPad] = static_cast<utils::bf16>(scales[i * N + j]);
                }
              }
              if (zero_points != nullptr) {
                std::memcpy(stor->template ZPtr<int8_t>() + i * stor->mNPad, zero_points + i * N,
                            N * sizeof(zero_points[0]));
              }
            } else {
              if (scales != nullptr)
                std::memset(stor->template SPtr<utils::bf16>() + i * stor->mNPad, 0, stor->mNPad * sizeof(utils::bf16));
              if (zero_points != nullptr)
                std::memset(stor->template ZPtr<int8_t>() + i * stor->mNPad, 0, stor->mNPad * sizeof(zero_points[0]));
            }
          }
        }
      });
    } else if (stor->SDtype() == BTLA_DTYPE::F8_E8M0) {
      threading->parallel_for([&](int tidx) {
        parallel::ThreadProblem2D thdp{tidx};
        _para.getIndex(thdp);
        if (thdp.valid) {
          for (int i = thdp.loc[1]; i < thdp.loc[1] + thdp.size[1]; i++) {
            if (i < rawnk_scale) {
              if (scales != nullptr) {
                for (int j = 0; j < N; j++) {
                  stor->template SPtr<utils::f8>()[j + i * stor->mNPad] = static_cast<int8_t>(scales[i * N + j]);
                }
              }
            } else {
              if (scales != nullptr)
                std::memset(stor->template SPtr<utils::f8>() + i * stor->mNPad, 0, stor->mNPad * sizeof(utils::f8));
            }
          }
        }
      });
    } else if (stor->SDtype() == BTLA_DTYPE::DQ8_BNB) {
      threading->parallel_for([&](int tidx) {
        parallel::ThreadProblem2D thdp{tidx};
        _para.getIndex(thdp);
        if (thdp.valid) {
          for (int i = thdp.loc[1]; i < thdp.loc[1] + thdp.size[1]; i++) {
            if (i < rawnk_scale) {
              if (scales != nullptr) {
                for (int j = 0; j < N; j++) {
                  stor->template SPtr<uint8_t>()[j + i * stor->mNPad] = static_cast<uint8_t>(scales[i * N + j]);
                }
              }
            } else {
              if (scales != nullptr)
                std::memset(stor->template SPtr<uint8_t>() + i * stor->mNPad, 0, stor->mNPad * sizeof(uint8_t));
            }
          }
        }
      });
    } else {
      assert(0);
    }
  }

  template <BTLA_ISA ISA_T, typename _T>
  static inline BTLA_CODE getWeight(_T** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                    const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = reinterpret_cast<StorageWeight*>(_param.packedW);
    auto NPad = wptr->mNPad;
    auto KPad = wptr->mKPad;
    char* bptr;
    if (wptr->mDType == BTLA_DTYPE::F8_E5M2 || wptr->mDType == BTLA_DTYPE::F8_E4M3) {
      bptr = wptr->template WPtr<char>() + n_offset * KPad + k_offset * _GemmCore_T::NTILE;
    } else {
      bptr = wptr->template WPtr<char>() + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2;
    }
    int constexpr ColSize = _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      if (wptr->SDtype() == BTLA_DTYPE::F8_E8M0) {
        assert(wptr->mDType == BTLA_DTYPE::F8_E4M3 || wptr->mDType == BTLA_DTYPE::F8_E5M2);
        auto sptr = wptr->template SPtr<utils::f8>() + n_offset + i;
        kernel::wrapper::DecompressKBlockF8FP<_GemmCore_T::PACK_ROW>::template forward<ISA_T>(
            reinterpret_cast<utils::f8*>(bptr) + i * KPad, *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW,
            ColSize, ColSize, ColSize, sptr, k_offset / _GemmCore_T::PACK_ROW, wptr->mBlockSize / _GemmCore_T::PACK_ROW,
            NPad, wptr->mDType);
      } else if (wptr->SDtype() == BTLA_DTYPE::F32) {
        auto sptr = wptr->template SPtr<float>() + n_offset + i;
        auto f4ptr = reinterpret_cast<utils::f4x2*>(bptr + i * KPad / 2);
        auto fp_ptr = *dstptr + i * k_size;
        if (wptr->mDType == BTLA_DTYPE::F8_E4M3 || wptr->mDType == BTLA_DTYPE::F8_E5M2) {
          kernel::wrapper::DecompressKBlockF8FP<_GemmCore_T::PACK_ROW>::template forward<ISA_T>(
              reinterpret_cast<utils::f8*>(bptr) + i * KPad, *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW,
              ColSize, ColSize, ColSize, sptr, k_offset / _GemmCore_T::PACK_ROW,
              wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad, wptr->mDType);
        } else if (wptr->mDType == BTLA_DTYPE::F4_NF4) {
          kernel::wrapper::DecompressKBlockF4Fp<_T, _GemmCore_T::PACK_ROW>::template forward<ISA_T, float,
                                                                                             BTLA_DTYPE::F4_NF4>(
              f4ptr, fp_ptr, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, sptr,
              k_offset / _GemmCore_T::PACK_ROW, wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad, tmpcache, cachesize);
        } else if (wptr->mDType == BTLA_DTYPE::F4_E2M1) {
          kernel::wrapper::DecompressKBlockF4Fp<_T, _GemmCore_T::PACK_ROW>::template forward<ISA_T, float,
                                                                                             BTLA_DTYPE::F4_E2M1>(
              f4ptr, fp_ptr, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, sptr,
              k_offset / _GemmCore_T::PACK_ROW, wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad, tmpcache, cachesize);
        } else if (wptr->mDType == BTLA_DTYPE::F4_BNB) {
          kernel::wrapper::DecompressKBlockF4Fp<_T, _GemmCore_T::PACK_ROW>::template forward<ISA_T, float,
                                                                                             BTLA_DTYPE::F4_BNB>(
              f4ptr, fp_ptr, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, sptr,
              k_offset / _GemmCore_T::PACK_ROW, wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad, tmpcache, cachesize);
        } else {
          assert(0);
        }
      } else if (wptr->SDtype() == BTLA_DTYPE::BF16) {
        auto sptr = wptr->template SPtr<utils::bf16>() + n_offset + i;
        auto f4ptr = reinterpret_cast<utils::f4x2*>(bptr + i * KPad / 2);
        auto fp_ptr = *dstptr + i * k_size;
        if (wptr->mDType == BTLA_DTYPE::F4_NF4) {
          kernel::wrapper::DecompressKBlockF4Fp<_T, _GemmCore_T::PACK_ROW>::template forward<ISA_T, utils::bf16,
                                                                                             BTLA_DTYPE::F4_NF4>(
              f4ptr, fp_ptr, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, sptr,
              k_offset / _GemmCore_T::PACK_ROW, wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad, tmpcache, cachesize);
        } else if (wptr->mDType == BTLA_DTYPE::F4_E2M1) {
          kernel::wrapper::DecompressKBlockF4Fp<_T, _GemmCore_T::PACK_ROW>::template forward<ISA_T, utils::bf16,
                                                                                             BTLA_DTYPE::F4_E2M1>(
              f4ptr, fp_ptr, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, sptr,
              k_offset / _GemmCore_T::PACK_ROW, wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad, tmpcache, cachesize);
        } else if (wptr->mDType == BTLA_DTYPE::F4_BNB) {
          kernel::wrapper::DecompressKBlockF4Fp<_T, _GemmCore_T::PACK_ROW>::template forward<ISA_T, utils::bf16,
                                                                                             BTLA_DTYPE::F4_BNB>(
              f4ptr, fp_ptr, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, sptr,
              k_offset / _GemmCore_T::PACK_ROW, wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad, tmpcache, cachesize);
        } else {
          assert(0);
        }
      } else if (wptr->SDtype() == BTLA_DTYPE::DQ8_BNB) {
        auto f4ptr = reinterpret_cast<utils::f4x2*>(bptr + i * KPad / 2);
        auto fp_ptr = *dstptr + i * k_size;
        auto internal_n_offset = n_offset + i;
        auto internal_k_offset = k_offset / _GemmCore_T::PACK_ROW;
        auto internal_kblock = wptr->mBlockSize / _GemmCore_T::PACK_ROW;
        auto dq_offset_idx = static_cast<int>(wptr->mCorrection.mDQCorrectionBuf.mBufSize / sizeof(float) - 1);
        if (wptr->mDType == BTLA_DTYPE::F4_NF4) {
          kernel::wrapper::DecompressDqKBlockF4Fp<_T, _GemmCore_T::PACK_ROW>::template forward<ISA_T,
                                                                                               BTLA_DTYPE::F4_NF4>(
              f4ptr, fp_ptr, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, wptr->template SPtr<uint8_t>(),
              wptr->template DQPtr<float>(), internal_k_offset, internal_n_offset, internal_kblock, wptr->mDqBlockSize,
              dq_offset_idx, NPad, wptr->mN, tmpcache, cachesize);
        } else {
          assert(0);
        }
      } else {
        assert(0);
      }
    }
    *dststep = k_size;
    return BTLA_CODE::Success;
  }

 protected:
  AUTOCALL void quantRowBlock(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst,
                              float* scales, StorageWeight* ptr) {
    auto quant_dtype = ptr->mDType;
    if (quant_dtype == BTLA_DTYPE::F8_E4M3) {
      kernel::wrapper::QuantizeF8RowBlock<BTLA_DTYPE::F8_E4M3>::forward_auto(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                                             scales, ptr->mBlockSize, ptr->SDtype());
    } else if (quant_dtype == BTLA_DTYPE::F8_E5M2) {
      kernel::wrapper::QuantizeF8RowBlock<BTLA_DTYPE::F8_E5M2>::forward_auto(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                                             scales, ptr->mBlockSize, ptr->SDtype());
    } else if (quant_dtype == BTLA_DTYPE::F4_BNB) {
      kernel::wrapper::QuantizeF4RowBlock<BTLA_DTYPE::F4_BNB>::forward_auto(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                                            scales, ptr->mBlockSize);
    } else if (quant_dtype == BTLA_DTYPE::F4_E2M1) {
      kernel::wrapper::QuantizeF4RowBlock<BTLA_DTYPE::F4_E2M1>::forward_auto(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                                             scales, ptr->mBlockSize);
    } else if (quant_dtype == BTLA_DTYPE::F4_NF4) {
      kernel::wrapper::QuantizeF4RowBlock<BTLA_DTYPE::F4_NF4>::forward_auto(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                                            scales, ptr->mBlockSize);
    } else {
      assert(0);
    }
  }
};
}  // namespace gemm
}  // namespace prologue_b
}  // namespace bestla
