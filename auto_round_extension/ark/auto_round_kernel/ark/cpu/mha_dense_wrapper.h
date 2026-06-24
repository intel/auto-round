//  Copyright (c) 2026 Intel Corporation
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

// -----------------------------------------------------------------------------
// ARK CPU flash-attention wrapper.
//
// This is a direct port of Neural Speed's BestLA attention wrapper
// (neural_speed/core/layers/mha_dense_wrapper.h) adapted to the BestLA snapshot
// vendored under auto_round_kernel/bestla. The eventual target is to land
// `mha_stable_interface_t` plus the `bestla_fusion_attn_forward` dtype
// specializations as the CPU SDPA runtime; the legacy scalar kernel in
// mha_dense.cpp is retained only as a temporary build-safety fallback and is not
// the long-term path.
//
// Phase 2, step 1 migrates the BestLA-independent softmax/epilogue building
// blocks that the stable interface composes:
//   * mha_exp_ref
//   * scale_write_back_t
//   * scale_track_max_t
//   * inplace_precompute_max_softmax_t
//   * activation_identity_t
//   * weight_base_t
//
// Phase 2, step 2 migrates the GEMM dispatch / packer layer that the stable
// interface launchers compose (kept in Neural Speed priority order):
//   * launcher_base_weight_t          (LauncherBase + N-dim track-max kernels)
//   * launcher_base_off_t             (LauncherBase + packed-weight batch offset)
//   * storage_packed_weight_batch_t   (batched packed-weight storage object)
//   * weight_pack_batch_bf16_base_t   (runtime bf16 weight packer base)
//   * weight_pack_batch_bf16_trans_t  (transposed source variant)
//   * weight_pack_batch_bf16_non_tr_t (non-transposed source variant)
//   * weight_forward_n_tile48_t       (NTILE=48 already-laid-out weight prologue)
//   * weight_cvt_bf16_ntile48_t       (bf16->dst NTILE=48 weight conversion)
//   * weight_cvt_f16_n_tile24_t       (fp16->fp32 NTILE=24 weight conversion)
// Runtime dispatch is intentionally NOT wired here; this step only lands the
// reusable launcher/prologue/packer building blocks. The next step is
// `mha_stable_interface_t`.
//
// API-drift notes vs Neural Speed's BestLA:
//   * ARK's `kernel::wrapper::ScaleTrackMax::forward` takes an extra
//     `padding_type` argument (0=dense, 1=causal, 2=right-padding) that Neural
//     Speed folds into `causal_offset`. We surface it as `scale_track_max_t::
//     Param::padding_type` (default 0) so both the causal and right-padding
//     routes can be driven later without re-touching the call site.
//   * Neural Speed gates `exp` behind the `MHA_2ND_EXP` macro; we mirror it but
//     default it on to reuse BestLA's `kernel::ref::exp_ps_0_1`.
//   * Neural Speed sizes the packed-weight buffer with `utils::bestla_dtype_size`
//     and aligns storage to the `NE_ALIGNMENT` macro. ARK's vendored BestLA
//     exposes neither; we use `utils::bestla_dtype_bytes` and the
//     `bestla::storage::Alignment` (== 64) constant instead. See
//     `storage_packed_weight_batch_t`.
//   * Neural Speed's wrapper relies on `using namespace bestla` to reach the
//     `padto / padto_le / remainsize / cpu_pointer_align` helpers and the
//     `bf16 / fp16` types unqualified. In ARK these live under `bestla::utils`,
//     so the launcher/packer bodies qualify them with `utils::` (and use
//     `utils::bf16 / utils::fp16`). Logic is otherwise byte-for-byte.
//   * ARK's `wrapper::gemm::LauncherBase` adds a `GEMVWrapper` fast path inside
//     its own `run()`. Our launchers fully override `run()/run_block()` (as in
//     Neural Speed) so that GEMV path is bypassed; only the member typedefs
//     (`GemmCore/Param/AType/BType/CType/ISA/PrologueA/PrologueB/Epilogue`) are
//     inherited, all of which the ARK base exposes under the same names.
// -----------------------------------------------------------------------------

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <type_traits>

#include "bestla/bestla.h"
#include "bestla/bestla_gemm.h"
#include "bestla/bestla_parallel.h"
#include "bestla/bestla_storage.h"
#include "bestla/bestla_utils.h"
#include "bestla/bestla_wrapper.h"
#include "bestla/kernel_ref.h"
#include "bestla/kernel_wrapper.h"

#include "mha_dense.h"

namespace ark::cpu {

// ---------------------------------------------------------------------------
// Legacy scalar helpers (used by the temporary mha_dense.cpp / sdpa.cpp path).
// These remain until the BestLA stable interface fully replaces the runtime.
// ---------------------------------------------------------------------------
size_t element_size(BTLA_DTYPE dtype);
float load_scalar(const void* base, size_t element_offset, BTLA_DTYPE dtype);
void store_scalar(void* base, size_t element_offset, BTLA_DTYPE dtype, float value);

// ---------------------------------------------------------------------------
// Neural-Speed-style BestLA attention components (Phase 2 migration).
// Namespace mirrors Neural Speed's ne_bestla::custom::mha grouping.
// ---------------------------------------------------------------------------
namespace bestla_mha {

using namespace bestla;  // NOLINT(build/namespaces): match Neural Speed wrapper

// Prefer BestLA's fast polynomial exp (matches Neural Speed's MHA_2ND_EXP path).
#ifndef ARK_MHA_2ND_EXP
#define ARK_MHA_2ND_EXP 1
#endif

inline float mha_exp_ref(float x) {
#if ARK_MHA_2ND_EXP
  return bestla::kernel::ref::exp_ps_0_1(x);
#else
  return std::exp(x);
#endif
}

/**
 * @brief Epilogue that scales the fp32 GEMM result (optionally per-row), casts
 * to the destination type and writes it back. Pure scalar; no ISA dependency.
 */
template <typename T_SRC, typename T_DST>
class scale_write_back_t {
 public:
  using SType = T_SRC;
  using DType = T_DST;
  struct Param {  // NOLINT(readability-identifier-naming): align with bestla name
    const float* scale;
    DType* dst;
    int ld_dst;
  };
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(const SType* src, const int src_step, const int M_offset, const int N_offset,
                                  const int M, const int N, const Param& p, void* /* tmpcache */,
                                  size_t /* cachesize */) {
    const auto dst = p.dst + M_offset * p.ld_dst + N_offset;
    const auto scale = p.scale + M_offset;
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j)  //
        dst[i * p.ld_dst + j] = static_cast<DType>(scale[i] * src[i * src_step + j]);
    return BTLA_CODE::Success;
  }
};
using ScaleWriteBackFp32Bf16 = scale_write_back_t<float, utils::bf16>;
using ScaleWriteBackFp32Fp32 = scale_write_back_t<float, float>;
using ScaleWriteBackS32S8 = scale_write_back_t<int32_t, int8_t>;

/**
 * @brief Epilogue for the QK matmul: scales the scores, applies the causal /
 * right-padding mask and tracks the per-row running max (the m_i of the
 * flash-attention stable softmax).
 *
 * Adapts to ARK's BestLA `ScaleTrackMax::forward`, which carries an explicit
 * `padding_type` argument (see file header). `Param::padding_type` defaults to
 * dense (0); callers set 1 for causal or 2 for right padding.
 */
template <typename T_SRC, typename T_DST>
class scale_track_max_t {
 public:
  using DType = T_DST;
  using SType = T_SRC;
  struct Param {  // NOLINT(readability-identifier-naming): align with bestla name
    DType* dst;
    DType* dst_max;
    int ld_dst;        // #elements
    float scale;
    int causal_offset;  // offset for causal mask; negative disables causal mask
    float alibi_slope;  // m-factor in the alibi paper (https://arxiv.org/abs/2108.12409)
    float tanh_scale;
    int padding_type = 0;  // ARK BestLA: 0=dense, 1=causal, 2=right-padding
  };
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(const SType* src, const int src_step, const int M_offset, const int N_offset,
                                  const int M, const int N, const Param& p, void* tmpcache, size_t cachesize) {
    return bestla::kernel::wrapper::ScaleTrackMax<SType, DType>::template forward<ISA_T>(
        src, src_step, p.dst, p.dst_max, p.ld_dst, M_offset, N_offset, M, N, p.scale, p.causal_offset, p.alibi_slope,
        p.tanh_scale, p.padding_type, tmpcache, cachesize);
  }
};
using ScaleTrackMaxFp16Fp32 = scale_track_max_t<utils::fp16, float>;
using ScaleTrackMaxFp32Fp32 = scale_track_max_t<float, float>;
using ScaleTrackMaxS32Fp32 = scale_track_max_t<int32_t, float>;

/**
 * @brief In-place stable softmax over the score tile: subtracts the row max,
 * exponentiates, and accumulates the per-row exp-sum (the l_i of flash
 * attention). Delegates to BestLA's vectorized kernel.
 */
template <class SRC_T, class DST_T>
struct inplace_precompute_max_softmax_t {
  // n_size is the starting n-size when the causal mask is enabled.
  // src and dst may alias when sizeof(SRC_T) >= sizeof(DST_T) and ld is set.
  // s_max and expsum may alias.
  template <BTLA_ISA ISA_T>
  static inline void forward(int m_size, int n_size, int n_pad_size, bool is_causal, SRC_T* src, DST_T* dst,
                             const SRC_T* s_max, float* expsum, int ld_src, int ld_dst) {
    const auto ret = bestla::kernel::wrapper::InplacePrecomputeMaxSoftmax<SRC_T, DST_T>::template forward<ISA_T>(
        m_size, n_size, n_pad_size, is_causal, src, dst, s_max, expsum, ld_src, ld_dst);
    assert(ret == BTLA_CODE::Success);
    (void)ret;
  }
};

/**
 * @brief Activation prologue that passes the A-matrix straight through (used for
 * the already-laid-out P matrix of the PV matmul).
 */
template <class _GemmCore_T>
class activation_identity_t {
 public:
  using AType = typename _GemmCore_T::AType;
  struct Param {  // NOLINT(readability-identifier-naming): align with bestla name
    const AType* A;
    int lda;
  };
  activation_identity_t() = default;

  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE getActivation(AType** dstptr, int* dststep, const Param& _param, int m_size, int k_size,
                                        int m_offset, int k_offset, void* /* tmpcache */, size_t /* cachesize */) {
    (void)m_size;
    (void)k_size;
    auto aptr = const_cast<AType*>(_param.A);
    *dstptr = aptr + m_offset * _param.lda + k_offset;
    *dststep = _param.lda;
    return BTLA_CODE::Success;
  }
};

/**
 * @brief Batched packed-weight storage object (one packed K/V tensor per head).
 *
 * Direct port of Neural Speed's `storage_packed_weight_batch_t`. ARK BestLA
 * drift (see file header):
 *   * `utils::bestla_dtype_size` -> `utils::bestla_dtype_bytes`.
 *   * `NE_ALIGNMENT` macro -> `bestla::storage::Alignment` (== 64).
 * The aligned buffer / (de)serialization plumbing is identical to Neural Speed:
 * ARK's `storage::ObjectAlignedBuffer` and `storage::gemm::IWeightBase` expose
 * the same `resize / get<T> / serializeToBuffer / deserializeBuffer` surface.
 */
class storage_packed_weight_batch_t : public storage::gemm::IWeightBase {
  using Base = storage::gemm::IWeightBase;

 public:
  int mBatch;
  storage::ObjectAlignedBuffer<storage::Alignment> mWBuf;
  // size_t mWSize;

  explicit storage_packed_weight_batch_t(uint64_t _core_id) : Base(_core_id), mBatch(0) {}
  size_t resize(int NPad, int KPad, int N, int K, int num_batch, BTLA_DTYPE dtype) {
    IWeightBase::resize(NPad, KPad, N, K, dtype);
    mBatch = num_batch;
    // ARK drift: Neural Speed uses utils::bestla_dtype_size here.
    auto bsize = static_cast<size_t>(mBatch) * NPad * KPad * utils::bestla_dtype_bytes(dtype);
    mWBuf.resize(bsize);
    // ARK drift: Neural Speed pads to the NE_ALIGNMENT macro.
    mSize = utils::padto(IWeightBase::getSerializedSize() + mWBuf.getSerializedSize(), storage::Alignment);
    return mSize;
  }

  template <typename T>
  inline constexpr T* WPtr() const {
    return mWBuf.get<T>();
  }

  void assign(int8_t* buf) override {
    deserializeBuffer(buf, true);
    mWBuf.deserializeBuffer(buf, true);
  }

  void serialize(int8_t* wptr) override {
    serializeToBuffer(wptr);
    mWBuf.serializeToBuffer(wptr);
  }

  void deserialize(int8_t* rptr) override {
    deserializeBuffer(rptr, false);
    mWBuf.deserializeBuffer(rptr, false);
  }

 protected:
  size_t getSerializedSize() override { return Base::getSerializedSize() + sizeof(mBatch); }

  void serializeToBuffer(int8_t*& wptr) override {
    Base::serializeToBuffer(wptr);
    utils::serialize(wptr, mBatch);
  }
  void deserializeBuffer(int8_t*& rptr, bool map_buf) override {
    Base::deserializeBuffer(rptr, map_buf);
    if (!map_buf) {
      mBatch = utils::deserialize<int>(rptr);
    } else {
      utils::serialize<int>(rptr, mBatch);
    }
  }
};

/**
 * @brief Weight prologue that packs Bf16 weight at runtime; base type shared by
 * the transposed / non-transposed source variants. Port of Neural Speed's
 * `weight_pack_batch_bf16_base_t`. Relies only on the packed-weight `mKPad /
 * mNPad` strides plus the GemmCore `NTILE / KTILE`, so it is layout-agnostic.
 */
template <class GemmCore_T, bool IsTrans, typename T_SRC = typename GemmCore_T::BType>
class weight_pack_batch_bf16_base_t {
 public:
  using WType = typename GemmCore_T::BType;           // weight type
  using SType = T_SRC;                                // source type (before packed)
  using StorageType = storage_packed_weight_batch_t;  // packed weight type

  struct Param {  // NOLINT(readability-identifier-naming): align with bestla name
    const SType* B;
    const int ldb;
    const StorageType* packedW;
  };

  TLACALL BTLA_CODE getWeight(...) = delete;

  TLACALL BTLA_CODE getWeight(WType** dstptr, int* dststep, int /* b_size */, int /* k_size */, int /* n_size */,
                              int b_offset, int k_offset, int n_offset, const Param& param, void* /* tmpcache */,
                              size_t /* cachesize */) {
    const auto wptr = param.packedW;
    if (!wptr) return BTLA_CODE::InvalidParam;
    assert(k_offset % GemmCore_T::KTILE == 0);
    assert(n_offset % GemmCore_T::NTILE == 0);
    auto KPad = wptr->mKPad;
    auto NPad = wptr->mNPad;
    (void)NPad;
    *dstptr = wptr->template WPtr<WType>() + n_offset * KPad + k_offset * GemmCore_T::NTILE;
    *dststep = KPad;
    return BTLA_CODE::Success;
  }

  TLACALL BTLA_CODE getWeight(WType** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                              const Param& param, void* tmpcache, size_t cachesize) {
    return getWeight<ISA_T>(dstptr, dststep, 1, k_size, n_size, 0, k_offset, n_offset, param, tmpcache, cachesize);
  }

  TLACALL BTLA_CODE packWeight(...) = delete;
};

/**
 * @brief Runtime bf16 weight packer for a transposed source (K-major). Port of
 * Neural Speed's `weight_pack_batch_bf16_trans_t`. Uses ARK BestLA's
 * `kernel::wrapper::PaddingTransInterleaveMN` (identical signature).
 */
template <class GemmCore_T, typename T_SRC = typename GemmCore_T::BType>
class weight_pack_batch_bf16_trans_t : public weight_pack_batch_bf16_base_t<GemmCore_T, true, T_SRC> {
  using Base = weight_pack_batch_bf16_base_t<GemmCore_T, true, T_SRC>;

 public:
  using typename Base::Param;
  using typename Base::StorageType;
  using typename Base::SType;
  using typename Base::WType;

  /// Reorder job of a thread
  AUTOCALL void run(const Param& p, const parallel::ThreadProblem2D& thdp, const std::function<int(int)>& step_batch) {
    if (!thdp.valid) return;
    const auto pw = dynamic_cast<const StorageType*>(p.packedW);
    assert(pw != nullptr);
    const int KPad = pw->mKPad;  // K size after transpose & padding
    const int NPad = pw->mNPad;  // N size after transpose & padding
    assert(pw->mK <= KPad);
    assert(pw->mN <= NPad);

    // y for batch; x for major-dim of the source data (N-dim of the packed weight)
    const auto [y, x] = thdp.loc;
    const auto [ny, nx] = thdp.size;
    const auto nx_pad = utils::padto(nx, GemmCore_T::NTILE);

    assert(utils::padto(pw->mK, GemmCore_T::KTILE) == KPad);

    using KernInterleave = typename kernel::wrapper::PaddingTransInterleaveMN<  //
        GemmCore_T::NTILE, GemmCore_T::PACK_ROW, T_SRC, WType>;

    for (int ibat = y; ibat < y + ny; ++ibat) {
      const auto forward_stat = KernInterleave::forward_auto(          //
          p.B + step_batch(ibat) + x * p.ldb,                          //
          pw->template WPtr<WType>() + ibat * KPad * NPad + x * KPad,  //
          nx, pw->mK,                                                  // size
          nx_pad, KPad,                                                // padded size
          p.ldb, KPad);                                                // step
      assert(forward_stat == BTLA_CODE::Success);
      (void)forward_stat;
    }
  }
};

/**
 * @brief Runtime bf16 weight packer for a non-transposed source (N-major). Port
 * of Neural Speed's `weight_pack_batch_bf16_non_tr_t`. Uses ARK BestLA's
 * `kernel::wrapper::PaddingInterleaveMN` (identical signature).
 */
template <class GemmCore_T, typename T_SRC = typename GemmCore_T::BType>
class weight_pack_batch_bf16_non_tr_t : public weight_pack_batch_bf16_base_t<GemmCore_T, false, T_SRC> {
  using Base = weight_pack_batch_bf16_base_t<GemmCore_T, false, T_SRC>;

 public:
  using typename Base::Param;
  using typename Base::StorageType;
  using typename Base::SType;
  using typename Base::WType;

  /// Reorder job of a thread
  AUTOCALL void run(const Param& p, const parallel::ThreadProblem2D& thdp, const std::function<int(int)>& step_batch) {
    if (!thdp.valid) return;
    const auto pw = dynamic_cast<const StorageType*>(p.packedW);
    assert(pw != nullptr);
    const int KPad = pw->mKPad;  // K size after padding
    const int NPad = pw->mNPad;  // N size after padding
    assert(pw->mK <= KPad);
    assert(pw->mN <= NPad);
    assert(utils::padto(pw->mN, GemmCore_T::NTILE) == NPad);

    auto [y, x] = thdp.loc;
    auto [ny, nx] = thdp.size;
    const auto nx_pad = utils::padto(nx, GemmCore_T::KTILE);
    (void)nx_pad;

    using KernInterleave = typename kernel::wrapper::PaddingInterleaveMN<  //
        GemmCore_T::NTILE, GemmCore_T::PACK_ROW, T_SRC, WType>;

    for (int ibat = y; ibat < y + ny; ++ibat) {
      const auto forward_stat = KernInterleave::forward_auto(                       //
          p.B + step_batch(ibat) + x * p.ldb,                                       //
          pw->template WPtr<WType>() + ibat * KPad * NPad + x * GemmCore_T::NTILE,  //
          nx, pw->mN,                                                               // size
          nx_pad, NPad,                                                             // padded size
          p.ldb, KPad);                                                             // stride
      assert(forward_stat == BTLA_CODE::Success);
      (void)forward_stat;
    }
  }
};

/**
 * @brief LauncherBase with an additional packed-weight offset input (used to
 * batch the K/V packed weights of a head). Port of Neural Speed's
 * `launcher_base_off_t`. Fully overrides `run()/run_block()` so ARK BestLA's
 * GEMV fast path in the base `run()` is bypassed (see file header). All helper
 * names (`padto / padto_le / remainsize / cpu_pointer_align`) are qualified with
 * `utils::` for ARK; the logic mirrors Neural Speed exactly.
 */
template <class _GemmCore_T, template <class> class _PrologueA_T, template <class> class _PrologueB_T,
          class _Epilogue_T>
class launcher_base_off_t                  //
    : public wrapper::gemm::LauncherBase<  //
          _GemmCore_T, _PrologueA_T, _PrologueB_T, _Epilogue_T> {
  using Base = wrapper::gemm::LauncherBase<  //
      _GemmCore_T, _PrologueA_T, _PrologueB_T, _Epilogue_T>;

 public:
  using typename Base::GemmCore;
  using Param = typename Base::Param;
  using AType = typename Base::AType;
  using BType = typename Base::BType;
  using CType = typename Base::CType;
  static constexpr auto RT_ISA = Base::ISA;

  static void run(const Param& _param, const parallel::gemm::ThreadProblemBase& _config,
                  int w_offset /* weight offset for batching */) {
    // Temporarily configure to max tiling size (matches Neural Speed).
    Base::GemmCore::configure(16, 16, 16);
    auto StackTmp = alloca(_config.stacksize);
    auto tmpB = reinterpret_cast<BType*>(StackTmp);
    tmpB = utils::cpu_pointer_align(tmpB);
    auto tmpA = reinterpret_cast<AType*>(tmpB + static_cast<size_t>(_config.block[1]) * _config.block[2]);
    tmpA = utils::cpu_pointer_align(tmpA);
    auto tmpC = reinterpret_cast<CType*>(tmpA + static_cast<size_t>(GemmCore::MTILE) * _config.block[2]);
    tmpC = utils::cpu_pointer_align(tmpC);
    auto tmpCache = tmpC + _config.block[0] * _config.block[1];
    tmpCache = utils::cpu_pointer_align(tmpCache);

    for (int itern = 0; itern < _config.size[1]; itern += _config.block[1]) {
      int n_remain = utils::remainsize(itern, _config.size[1], _config.block[1]);
      for (int iterm = 0; iterm < _config.size[0]; iterm += _config.block[0]) {
        int m_remain = utils::remainsize(iterm, _config.size[0], _config.block[0]);
        run_block(_param, _config, w_offset, iterm, itern, m_remain, n_remain, tmpA, tmpB, tmpC, tmpCache);
      }
    }
  }

 protected:
  static void run_block(const Param& _param, const parallel::gemm::ThreadProblemBase& _config,
                        int w_offset /* weight offset for batching */, int blk_m, int blk_n, int blk_msize,
                        int blk_nsize, AType* tmpA, BType* /*tmpB*/, CType* tmpC, void* tmpcache) {
    int n_padded = utils::padto(blk_nsize, GemmCore::NTILE);
    for (int iterk = 0; iterk < _param.problem.dims[3]; iterk += _config.block[2]) {
      int k_remain = utils::remainsize(iterk, _param.problem.dims[3], _config.block[2]);
      int k_padded = utils::padto(k_remain, GemmCore::KTILE);
      int k_paddedle = utils::padto_le(k_remain, GemmCore::KTILE);
      BType* bptr_cache = nullptr;
      int bcache_step = 0;
      Base::PrologueB::template getWeight<Base::ISA>(&bptr_cache, &bcache_step,    //
                                                     k_padded, n_padded,           //
                                                     iterk, _config.loc[1] + blk_n,  //
                                                     _param.paramB, tmpcache, _config.tmpcachesize);
      bptr_cache += w_offset;
      int bcache_stride = bcache_step * sizeof(BType);
      for (int i = 0; i < blk_msize; i += GemmCore::MTILE) {
        int m_remain = utils::remainsize(i, blk_msize, GemmCore::MTILE);
        auto cptr_cache = tmpC + i * _config.block[1];
        int ccache_stride = _config.block[1] * sizeof(CType);

        int acache_step = 0;
        if (k_paddedle) {
          AType* aptr_cache = tmpA;
          Base::PrologueA::template getActivation<Base::ISA>(&aptr_cache, &acache_step, _param.paramA, m_remain,
                                                             k_paddedle, blk_m + i + _config.loc[0], iterk, tmpcache,
                                                             _config.tmpcachesize);
          Base::GemmCore::forward(aptr_cache, bptr_cache, cptr_cache, m_remain, n_padded, k_paddedle,
                                  acache_step * sizeof(AType), bcache_stride, ccache_stride, iterk, tmpcache,
                                  _config.tmpcachesize);
        }
        int k_tail = k_remain - k_paddedle;
        if (k_tail) {
          AType* aptr_cache = tmpA;
          Base::PrologueA::template getActivation<Base::ISA>(&aptr_cache, &acache_step, _param.paramA, m_remain, k_tail,
                                                             blk_m + i + _config.loc[0], iterk + k_paddedle, tmpcache,
                                                             _config.tmpcachesize);
          Base::GemmCore::forward(aptr_cache, bptr_cache + k_paddedle * GemmCore::NTILE, cptr_cache, m_remain, n_padded,
                                  GemmCore::KTILE, acache_step * sizeof(AType), bcache_stride, ccache_stride,
                                  iterk + k_paddedle, tmpcache, _config.tmpcachesize);
        }
      }
    }
    Base::Epilogue::template forward<Base::ISA>(tmpC, _config.block[1], _config.loc[0] + blk_m, _config.loc[1] + blk_n,
                                                blk_msize, blk_nsize, _param.paramC, tmpcache, _config.tmpcachesize);
  }
};

/**
 * @brief LauncherBase variant for the N-dim-parallel track-max QK / scaled PV
 * matmuls. Port of Neural Speed's `launcher_base_weight_t`. Same override
 * rationale and `utils::` qualification as `launcher_base_off_t`.
 */
template <class _GemmCore_T, template <class> class _PrologueA_T, template <class> class _PrologueB_T,
          class _Epilogue_T>
class launcher_base_weight_t               //
    : public wrapper::gemm::LauncherBase<  //
          _GemmCore_T, _PrologueA_T, _PrologueB_T, _Epilogue_T> {
  using Base = wrapper::gemm::LauncherBase<  //
      _GemmCore_T, _PrologueA_T, _PrologueB_T, _Epilogue_T>;

 public:
  using typename Base::GemmCore;
  using Param = typename Base::Param;
  using AType = typename Base::AType;
  using BType = typename Base::BType;
  using CType = typename Base::CType;
  static constexpr auto RT_ISA = Base::ISA;

  static void run(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
    Base::GemmCore::configure(16, 16, 16);
    auto StackTmp = alloca(_config.stacksize);
    auto tmpB = reinterpret_cast<BType*>(StackTmp);
    tmpB = utils::cpu_pointer_align(tmpB);
    auto tmpA = reinterpret_cast<AType*>(tmpB + static_cast<size_t>(_config.block[1]) * _config.block[2]);
    tmpA = utils::cpu_pointer_align(tmpA);
    auto tmpC = reinterpret_cast<CType*>(tmpA + static_cast<size_t>(GemmCore::MTILE) * _config.block[2]);
    tmpC = utils::cpu_pointer_align(tmpC);
    auto tmpCache = tmpC + _config.block[0] * _config.block[1];
    tmpCache = utils::cpu_pointer_align(tmpCache);

    for (int itern = 0; itern < _config.size[1]; itern += _config.block[1]) {
      int n_remain = utils::remainsize(itern, _config.size[1], _config.block[1]);
      for (int iterm = 0; iterm < _config.size[0]; iterm += _config.block[0]) {
        int m_remain = utils::remainsize(iterm, _config.size[0], _config.block[0]);
        run_block(_param, _config, iterm, itern, m_remain, n_remain, tmpA, tmpB, tmpC, tmpCache);
      }
    }
  }

 protected:
  static void run_block(const Param& _param, const parallel::gemm::ThreadProblemBase& _config, int blk_m, int blk_n,
                        int blk_msize, int blk_nsize, AType* tmpA, BType* tmpB, CType* tmpC, void* tmpcache) {
    int n_padded = utils::padto(blk_nsize, GemmCore::NTILE);
    for (int iterk = 0; iterk < _param.problem.dims[3]; iterk += _config.block[2]) {
      int k_remain = utils::remainsize(iterk, _param.problem.dims[3], _config.block[2]);
      int k_padded = utils::padto(k_remain, GemmCore::KTILE);
      int k_paddedle = utils::padto_le(k_remain, GemmCore::KTILE);
      auto bptr_cache = tmpB;
      int bcache_step = 0;

      Base::PrologueB::template getWeight<Base::ISA>(&bptr_cache, &bcache_step, _param.paramB, k_padded, blk_nsize,
                                                     iterk, _config.loc[1] + blk_n, tmpcache, _config.tmpcachesize);
      int bcache_stride = bcache_step * sizeof(BType);
      for (int i = 0; i < blk_msize; i += GemmCore::MTILE) {
        int m_remain = utils::remainsize(i, blk_msize, GemmCore::MTILE);
        auto cptr_cache = tmpC + i * _config.block[1];
        int ccache_stride = _config.block[1] * sizeof(CType);

        int acache_step = 0;
        if (k_paddedle) {
          AType* aptr_cache = tmpA;
          Base::PrologueA::template getActivation<Base::ISA>(&aptr_cache, &acache_step, _param.paramA, m_remain,
                                                             k_paddedle, (blk_m + i + _config.loc[0]), iterk, tmpcache,
                                                             _config.tmpcachesize);
          Base::GemmCore::forward(aptr_cache, bptr_cache, cptr_cache, m_remain, n_padded, k_paddedle,
                                  acache_step * sizeof(AType), bcache_stride, ccache_stride, iterk, tmpcache,
                                  _config.tmpcachesize);
        }
        int k_tail = k_remain - k_paddedle;
        if (k_tail) {
          AType* aptr_cache = tmpA;
          Base::PrologueA::template getActivation<Base::ISA>(&aptr_cache, &acache_step, _param.paramA, m_remain, k_tail,
                                                             (blk_m + i + _config.loc[0]), iterk + k_paddedle, tmpcache,
                                                             _config.tmpcachesize);
          Base::GemmCore::forward(aptr_cache, bptr_cache + k_paddedle * GemmCore::NTILE, cptr_cache, m_remain, n_padded,
                                  GemmCore::KTILE, acache_step * sizeof(AType), bcache_stride, ccache_stride,
                                  iterk + k_paddedle, tmpcache, _config.tmpcachesize);
        }
      }
    }
    Base::Epilogue::template forward<Base::ISA>(tmpC, _config.block[1], (_config.loc[0] + blk_m),
                                                _config.loc[1] + blk_n, blk_msize, blk_nsize, _param.paramC, tmpcache,
                                                _config.tmpcachesize);
  }
};

/**
 * @brief Weight prologue that exposes a plain (row-major) B matrix to the GEMM,
 * padding the N dimension to the GemmCore NTILE when needed.
 */
template <class _GemmCore_T>
class weight_base_t {
 public:
  using BType = typename _GemmCore_T::BType;
  using SType = BType;
  struct Param {  // NOLINT(readability-identifier-naming): align with bestla name
    const SType* B;
    int ldb;
    bool is_padded;
  };
  weight_base_t() = default;
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE getWeight(BType** dst_ptr, int* dst_step, const Param& p, int k_size, int n_size,
                                    int k_offset, int n_offset, void* /* tmpcache */, size_t /* cachesize */) {
    if ((n_size % _GemmCore_T::NTILE == 0) && std::is_same<SType, BType>::value &&
        false) {  // TODO: use a gemm core that accepts a step for K, or reorder at runtime
      *dst_ptr = const_cast<SType*>(p.B) + k_offset * p.ldb + n_offset;
      *dst_step = p.ldb;
      return BTLA_CODE::Success;
    } else if (*dst_ptr != nullptr && std::is_same<SType, BType>::value) {
      const auto src = const_cast<SType*>(p.B) + k_offset * p.ldb + n_offset;
      const auto npad = utils::padto(n_size, _GemmCore_T::NTILE);
      *dst_step = npad;
      for (int k = 0; k < k_size; ++k) {
        std::memcpy(*dst_ptr + k * npad, src + k * p.ldb, sizeof(BType) * n_size);
        std::memset(*dst_ptr + k * npad + n_size, 0, sizeof(BType) * (npad - n_size));
      }
      return BTLA_CODE::Success;
    } else {
      assert(false);
      return BTLA_CODE::NotSupport;
    }
  }
};

/**
 * @brief Weight prologue for already-laid-out (NTILE=48) weights. Port of Neural
 * Speed's `weight_forward_n_tile48_t`. Pure pointer arithmetic; the `48` packed
 * column stride matches the GemmCore NTILE the QK/PV matmuls are configured with
 * and is kept verbatim from Neural Speed.
 */
template <class _GemmCore_T>
class weight_forward_n_tile48_t {
 public:
  using BType = typename _GemmCore_T::BType;
  using SType = BType;
  struct Param {  // NOLINT(readability-identifier-naming): align with bestla name
    const SType* B;
    int ldb;
    bool is_padded;
  };
  weight_forward_n_tile48_t() = default;
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE getWeight(BType** dst_ptr, int* dst_step, const Param& p, int k_size, int n_size,
                                    int k_offset, int n_offset, void* /* tmpcache */, size_t /* cachesize */) {
    (void)k_size;
    (void)n_size;
    assert(p.is_padded);
    *dst_ptr = const_cast<SType*>(p.B) + k_offset * 48 + n_offset * p.ldb;
    *dst_step = p.ldb;
    return BTLA_CODE::Success;
  }
};

/**
 * @brief Weight prologue that converts a bf16 weight to the GemmCore dst type on
 * the fly for an NTILE=48 layout. Port of Neural Speed's
 * `weight_cvt_bf16_ntile48_t`; delegates to ARK BestLA's
 * `kernel::wrapper::WeightCvtBf16Ntile48` (same forward signature, parameterised
 * by the destination type).
 */
template <class _GemmCore_T>
class weight_cvt_bf16_ntile48_t {
 public:
  using BType = typename _GemmCore_T::BType;
  using SType = utils::bf16;  // ARK drift: Neural Speed names this bare `bf16`.
  struct Param {  // NOLINT(readability-identifier-naming): align with bestla name
    const SType* B;
    int ldb;
    bool is_padded;
  };

  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE getWeight(BType** dst_ptr, int* dst_step, const Param& p, int k_size, int n_size,
                                    int k_offset, int n_offset, void* tmpcache, size_t cachesize) {
    assert(p.is_padded);
    *dst_step = _GemmCore_T::NTILE;
    return kernel::wrapper::WeightCvtBf16Ntile48<BType>::template forward<ISA_T>(
        p.B, p.ldb, p.is_padded, *dst_ptr, *dst_step, k_size, n_size, k_offset, n_offset, tmpcache, cachesize);
  }
};

/**
 * @brief Weight prologue that converts an fp16 weight to fp32 (via F16C) for an
 * NTILE=24 layout. Port of Neural Speed's `weight_cvt_f16_n_tile24_t`; delegates
 * to ARK BestLA's `kernel::wrapper::WeightCvtFp16Ntile24`.
 */
template <class _GemmCore_T>
class weight_cvt_f16_n_tile24_t {  // convert fp16 weight to fp32 using F16C
 public:
  using BType = typename _GemmCore_T::BType;
  using SType = utils::fp16;  // ARK drift: Neural Speed names this bare `fp16`.
  struct Param {  // NOLINT(readability-identifier-naming): align with bestla name
    const SType* B;
    int ldb;
    bool is_padded;
  };

  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE getWeight(BType** dst_ptr, int* dst_step, const Param& p, int k_size, int n_size,
                                    int k_offset, int n_offset, void* tmpcache, size_t cachesize) {
    return kernel::wrapper::WeightCvtFp16Ntile24<BType>::template forward<ISA_T>(
        p.B, p.ldb, p.is_padded, *dst_ptr, *dst_step, k_size, n_size, k_offset, n_offset, tmpcache, cachesize);
  }
};

// ---------------------------------------------------------------------------
// Concrete instantiations / syntax-checks against the ARK vendored BestLA cores.
// These mirror Neural Speed's *NonTr / *Trans aliases and pin each migrated
// template to at least one real GemmCore so the building blocks are compiled
// here rather than only when the stable interface is wired in a later step.
// ---------------------------------------------------------------------------
using PackedWeightBatch = storage_packed_weight_batch_t;

// bf16 packers on the AMX bf16 core (BType == utils::bf16).
template <class GEMM_T>
using WeightPackBatchBf16Bf16NonTr = weight_pack_batch_bf16_non_tr_t<GEMM_T, utils::bf16>;
template <class GEMM_T>
using WeightPackBatchBf16Bf16Trans = weight_pack_batch_bf16_trans_t<GEMM_T, utils::bf16>;
template <class GEMM_T>
using WeightPackBatchFp16Bf16NonTr = weight_pack_batch_bf16_non_tr_t<GEMM_T, utils::fp16>;
template <class GEMM_T>
using WeightPackBatchFp16Bf16Trans = weight_pack_batch_bf16_trans_t<GEMM_T, utils::fp16>;

namespace instantiation_check {
// AVX2 fp32 core (SCoreRowNAvx2<24, 4>): drives the fp16->fp32 N-tile-24 path.
using CoreAvx2 = gemm::SCoreRowNAvx2<24, 4>;
// AVX512F fp32 core (SCoreRowNAvx512f<48, 8>): drives the bf16->fp32 N-tile-48 path.
using CoreAvx512f = gemm::SCoreRowNAvx512f<48, 8>;
// AMX bf16 core (HCoreRowNAmxbf16<48, 16>): drives the bf16 batched packers.
using CoreAmxBf16 = gemm::HCoreRowNAmxbf16<48, 16>;

// Launchers composed exactly as the stable interface will compose them.
using LauncherWeightAvx512f =
    launcher_base_weight_t<CoreAvx512f, activation_identity_t, weight_cvt_bf16_ntile48_t, ScaleWriteBackFp32Fp32>;
using LauncherWeightAvx2 =
    launcher_base_weight_t<CoreAvx2, activation_identity_t, weight_cvt_f16_n_tile24_t, ScaleWriteBackFp32Fp32>;
using LauncherOffAvx512f =
    launcher_base_off_t<CoreAvx512f, activation_identity_t, WeightPackBatchBf16Bf16NonTr, ScaleWriteBackFp32Fp32>;
using LauncherOffAmxBf16 =
    launcher_base_off_t<CoreAmxBf16, activation_identity_t, WeightPackBatchBf16Bf16NonTr, ScaleWriteBackFp32Bf16>;

// Packers / forward prologue pinned to concrete cores.
using PackBf16NonTr = WeightPackBatchBf16Bf16NonTr<CoreAmxBf16>;
using PackBf16Trans = WeightPackBatchBf16Bf16Trans<CoreAmxBf16>;
using PackFp16NonTr = WeightPackBatchFp16Bf16NonTr<CoreAmxBf16>;
using PackFp16Trans = WeightPackBatchFp16Bf16Trans<CoreAmxBf16>;
using ForwardNTile48 = weight_forward_n_tile48_t<CoreAvx512f>;
using CvtBf16NTile48 = weight_cvt_bf16_ntile48_t<CoreAvx512f>;
using CvtFp16NTile24 = weight_cvt_f16_n_tile24_t<CoreAvx2>;
}  // namespace instantiation_check

}  // namespace bestla_mha
}  // namespace ark::cpu
