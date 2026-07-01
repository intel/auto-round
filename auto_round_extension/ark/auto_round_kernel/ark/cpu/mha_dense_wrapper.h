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
// Phase 2, step 3 migrates the stable-softmax attention interface itself:
//   * attn_fwd_args_t<Q_T, K_T, V_T, DST_T> (typed-pointer argument bundle that
//     the wrapper consumes; mirrors Neural Speed's templated wrapper struct)
//   * mha_stable_interface_t<L_Max, L_Scale> (PrologueQ/K/S/V, QK*/PV* arg
//     typedefs, GemmQK/GemmPV, M_TILE/RT_ISA, and the full `compute()` flash
//     attention launcher: QxK -> stable softmax -> PxV).
// Runtime dispatch (sdpa.cpp / ark.cpp) and the dtype-specialized
// `bestla_fusion_attn_forward` overloads are still NOT wired here; the
// `instantiation_check` namespace only pins the interface to concrete BestLA
// cores so it is type-checked / compiled at this step.
//
// Phase 2, step 4 migrates the dtype-specialized attention dispatch:
//   * bestla_fusion_attn_forward<Q_T, K_T, V_T, DST_T> (generic primary
//     template `= delete`, so unsupported operand-type combinations are
//     rejected at compile time).
//   * bestla_fusion_attn_forward<float, fp16, fp16, float> (AVX2 stable branch).
//   * bestla_fusion_attn_forward<float, bf16, bf16, float> (AVX512F + AMX-BF16
//     stable branches, gated by ATTN_FLAG_PREFER_FP32 like Neural Speed).
// Only the fp32-score routes that compose `mha_stable_interface_t` are wired;
// the bf16/bf16, fp16/fp16 and int8 overloads (and the AVX512-FP16 / AMX-BF16
// ExpSum sub-paths) need the not-yet-migrated non-stable `mha_interface_t` /
// `ScaleExpAccSumFp32Bf16` / avx512fp16 core and assert off as scaffolding.
// Runtime dispatch (sdpa.cpp / ark.cpp) still does NOT call these overloads.
//
// Phase 4.5, step 1 begins the homogeneous FP16/BF16 attention path (Q, K, V and
// dst all one low-precision element type), the next major missing functional
// block after the stable mixed-precision (fp32-score) closure and the packed KV
// infrastructure. Neural Speed implements it with the *non-stable*
// `mha_interface_t` (single-pass QK*V that folds the softmax denominator into the
// PV accumulation via an ExpSum epilogue) rather than the two-pass
// `mha_stable_interface_t` this file has migrated so far:
//   * bestla_fusion_attn_forward<fp16, fp16, fp16, fp16> drives BestLA's
//     `gemm::HCoreRowNAvx512fp16` (native fp16 A/B/C GemmCore, ISA AVX512-FP16)
//     with a `kernel::wrapper::ScaleExpAccSumFp32<fp16>` QK epilogue.
//   * bestla_fusion_attn_forward<bf16, bf16, bf16, bf16> drives the AMX-BF16
//     `gemm::HCoreRowNAmxbf16` core with a `ScaleExpAccSumFp32<bf16>` /
//     `ScaleExpAccSumFp32Bf16` QK epilogue (the `avx512_bf16` sub-path of
//     `ScaleExpAccSumFp32` migrated at kernel_wrapper.h).
// This step only lands the two homogeneous `bestla_fusion_attn_forward`
// specializations as documented throwing scaffolding (so the operand-type
// surface exists and unsupported ISA/layout dispatches fail loudly rather than
// via a hard `= delete` compile error) plus compile-only `instantiation_check`
// pins for the homogeneous GemmCores. The non-stable `mha_interface_t` launcher
// and its ExpSum epilogue composition are NOT migrated here, and runtime
// dispatch (sdpa.cpp / ark.cpp) still does NOT route to these overloads; both
// are deferred to the following Phase 4.5 steps, mirroring how the mixed
// overloads were first introduced as scaffolding in Phase 2 step 4.
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
//   * Neural Speed's stable interface only handles dense + causal masking. ARK
//     adds an `ATTN_FLAG_PADDING_RIGHT` route: when set, `compute()` clamps the
//     unmasked K/V region to `attn_fwd_args_t::n_padding` and drives the QK
//     epilogue with `scale_track_max_t::Param::padding_type = 2`
//     (`causal_offset = n_padding`). ARK's `ScaleTrackMax` ref/AVX2/AVX512F
//     paths implement padding_type 2; the int8/fp16 paths assert it off, so the
//     right-padding route is currently fp32-score only (scaffolding).
//   * Neural Speed reaches the running CPU device through `GetCPUDevice()` and a
//     `NS_TP_MODEL` tensor-parallel block. ARK keeps `GetCPUDevice()` (vendored
//     BestLA macro) but drops the TP block (`k_offset = 0`,
//     `log_head_num = head_num`); alibi slope math is otherwise identical.
//   * Neural Speed's wrapper struct is `ne_bestla::custom::mha::attn_fwd_args_t
//     <...>` with bare `ne_attn_flags_t`. ARK mirrors it as
//     `bestla_mha::attn_fwd_args_t<...>` using ARK's `attn_flags_t` /
//     `ATTN_FWD_LAYOUT` (from mha_dense.h) and adds the `n_padding` field. The
//     non-templated `ark::cpu::attn_fwd_args_t` (Phase 1, void* pointers) is the
//     public C-style ABI struct and is unrelated to this typed wrapper struct.
//   * Neural Speed's `bestla_fusion_attn_forward` overloads take no threading
//     argument and pull a process-global pool from `ne_threading::get()`. ARK
//     has no such global, so each overload takes an explicit
//     `parallel::IThreading&` (the object `mha_stable_interface_t::compute`
//     already consumes) and forwards it through.
// -----------------------------------------------------------------------------

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <type_traits>

#include "bestla/bestla.h"
#include "bestla/bestla_device.h"
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
 * @brief Typed-pointer argument bundle consumed by `mha_stable_interface_t`.
 *
 * Direct port of Neural Speed's `ne_bestla::custom::mha::attn_fwd_args_t<...>`.
 * Unlike the public, void*-erased `ark::cpu::attn_fwd_args_t` (Phase 1, in
 * mha_dense.h), this struct carries fully-typed Q/K/V/dst pointers so the
 * wrapper can do element-wise pointer arithmetic with the per-tensor `step_*`
 * strides. Layout is therefore stride-driven only: no concrete [B,H,N,D] /
 * [B,N,H,D] order is assumed here.
 *
 * ARK drift vs Neural Speed (see file header):
 *   * `ne_attn_flags_t` -> ARK `attn_flags_t`; `ATTN_FWD_LAYOUT` is shared.
 *   * Adds `n_padding` to drive the `ATTN_FLAG_PADDING_RIGHT` route.
 */
template <typename Q_T, typename K_T, typename V_T, typename DST_T>
struct attn_fwd_args_t {
  Q_T* Q;
  K_T* K;
  V_T* V;
  DST_T* dst;
  float Q_sc, K_sc, V_sc, dst_sc;
  char* tmp;
  float QK_scale;
  attn_flags_t attn_flags;
  int batch_size, head_num, heads_kv, head_size, sl_q, sl_kv;
  ATTN_FWD_LAYOUT Q_layout, K_layout, V_layout, dst_layout;
  int step_q_bs, step_q_head_num, step_q_sl;
  int step_k_bs, step_k_head_num, step_k_sl, step_k_head_size;
  int step_v_bs, step_v_head_num, step_v_sl, step_v_head_size;
  int step_dst_bs, step_dst_head_num, step_dst_sl;
  // Number of valid (non-padding) K/V positions when ATTN_FLAG_PADDING_RIGHT is
  // set (ARK addition; ignored otherwise).
  int n_padding = 0;
};

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

/**
 * @brief MHA interface with N-dim parallelism & stable (flash-attention)
 * softmax. Port of Neural Speed's `mha_stable_interface_t`.
 *
 * @tparam L_Max   Launcher of the QxK matmul; tracks the running per-row max
 *                 (the m_i of the stable softmax) via a `scale_track_max_t`
 *                 epilogue.
 * @tparam L_Scale Launcher of the PxV matmul; scales the accumulated output by
 *                 1/l_i (and the dequant scales) in its epilogue.
 *
 * Both launchers are `launcher_base_weight_t` (N-dim parallel). The `step_*`
 * stride interface is itself HND/NHD-friendly: HND ([B,H,N,D]) and NHD
 * ([B,N,H,D]) Q/dst are expressed purely through strides. The K/V operands,
 * however, are NOT raw-layout-agnostic in the wired paths: the prologues
 * consume packed/reordered (NTILE24/NTILE48 row-packed) K/V, so a raw PLAIN
 * HND/NHD K/V tensor is unsupported until packing is added in Phase 4. See the
 * file header for the ARK BestLA API drift this port absorbs (`COMPUTE` vs
 * `COMP`, dropped TP block, PADDING_RIGHT route).
 */
template <class L_Max, class L_Scale>
class mha_stable_interface_t {
  template <class EpiArgs, bool HAS_SCALE, class T>
  static inline typename std::enable_if<!HAS_SCALE, EpiArgs>::type composeEpiArgs(float*, T* dst, int ld_dst) {
    return {dst, ld_dst};
  }
  template <class EpiArgs, bool HAS_SCALE, class T>
  static inline typename std::enable_if<HAS_SCALE, EpiArgs>::type composeEpiArgs(float* scale, T* dst, int ld_dst) {
    return {scale, dst, ld_dst};
  }

 public:
  using PrologueQ = typename L_Max::PrologueA;
  using PrologueK = typename L_Max::PrologueB;
  using QKProQArgs = typename PrologueQ::Param;
  using QKProKArgs = typename PrologueK::Param;
  using QKArgs = typename L_Max::Param;
  using QKEpiArgs = typename L_Max::EpiParam;

  using PrologueS = typename L_Scale::PrologueA;
  using PrologueV = typename L_Scale::PrologueB;
  using PVProPArgs = typename PrologueS::Param;
  using PVProVArgs = typename PrologueV::Param;
  using PVArgs = typename L_Scale::Param;
  using PVEpiArgs = typename L_Scale::EpiParam;

  using GemmQK = typename L_Max::GemmCore;
  using GemmPV = typename L_Scale::GemmCore;
  using Q_T = typename std::remove_const<typename std::remove_pointer<decltype(QKProQArgs::A)>::type>::type;
  using K_T = typename PrologueK::SType;
  using V_T = typename PrologueV::SType;
  using DST_T = typename L_Scale::Epilogue::DType;

  static constexpr auto RT_ISA = std::max(L_Max::RT_ISA, L_Scale::RT_ISA);

  static_assert(GemmQK::MTILE == GemmPV::MTILE, "2 GEMM should have the same M_TILE.");
  static constexpr auto M_TILE = GemmQK::MTILE;

  BTLA_CODE compute(const attn_fwd_args_t<Q_T, K_T, V_T, DST_T>& p, parallel::IThreading& th) {
    assert((std::is_same<Q_T, int8_t>::value || p.Q_sc == 1));
    assert((std::is_same<K_T, int8_t>::value || p.K_sc == 1));
    assert((std::is_same<V_T, int8_t>::value || p.V_sc == 1));
    assert((std::is_same<DST_T, int8_t>::value || p.dst_sc == 1));

    assert((p.Q_layout == ATTN_FWD_LAYOUT_PLAIN && p.dst_layout == ATTN_FWD_LAYOUT_PLAIN));
    assert((p.K_layout == ATTN_FWD_LAYOUT_PLAIN ||
            (std::is_same<K_T, int8_t>::value && p.K_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4) ||
            (std::is_same<K_T, utils::bf16>::value && p.K_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2) ||
            (std::is_same<K_T, utils::fp16>::value && p.K_layout == ATTN_FWD_LAYOUT_NTILE24_ROWPACK1)));
    assert((p.V_layout == ATTN_FWD_LAYOUT_PLAIN ||
            (std::is_same<V_T, int8_t>::value && p.V_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4) ||
            (std::is_same<V_T, utils::bf16>::value && p.V_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2) ||
            (std::is_same<V_T, utils::fp16>::value && p.V_layout == ATTN_FWD_LAYOUT_NTILE24_ROWPACK1)));

    assert((!std::is_same<  //
               PrologueK, weight_forward_n_tile48_t<typename L_Max::GemmCore>>::value) ||
           p.K_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4 ||
           p.K_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2);  // WeightForward needs a preprocessed layout

    assert((!std::is_same<  //
               PrologueV, weight_forward_n_tile48_t<typename L_Scale::GemmCore>>::value) ||
           p.V_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4 ||
           p.V_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2);  // WeightForward needs a preprocessed layout

    assert((p.K_layout != ATTN_FWD_LAYOUT_PLAIN || p.step_v_head_size == 1));
    assert((p.V_layout != ATTN_FWD_LAYOUT_PLAIN || p.step_k_sl == 1));
    const auto num_heads = p.batch_size * p.head_num;  // Total number of heads
    GetCPUDevice();
    const bool is_causal = (p.attn_flags & ATTN_FLAG_IS_CAUSAL) != 0;
    const bool is_alibi = (p.attn_flags & ATTN_FLAG_IS_ALIBI8) != 0;  // only support alibi with 8 now
    const bool is_tanh = (p.attn_flags & ATTN_FLAG_IS_TANH30) != 0;   // only support tanh with 30 now
    const bool prefer_fp32 = (p.attn_flags & ATTN_FLAG_PREFER_FP32) != 0;
    // ARK addition: right-padded variable-length batch (see file header).
    const bool is_padding = (p.attn_flags & ATTN_FLAG_PADDING_RIGHT) != 0;

    // prefer_fp32 requires both GEMMs to be fp32 compute cores.
    assert(("prefer_fp32 not followed!",  //
            !prefer_fp32 || (GemmQK::COMP == bestla::gemm::CompType::COMP_FP32 &&
                             GemmPV::COMP == bestla::gemm::CompType::COMP_FP32)));
    (void)prefer_fp32;
    assert(("qlen should be no greater then klen/vlen!", !is_causal || p.sl_q <= p.sl_kv));
    assert(!is_causal || p.sl_q <= p.sl_kv);
    assert(("head_num must be a multiple of heads_kv!", p.head_num % p.heads_kv == 0));
    const auto group_heads = p.head_num / p.heads_kv;  // GQA: ihkv = ihn / group_heads
    const auto sl_diff = p.sl_kv - p.sl_q;
    // ARK addition: number of valid K/V positions for the right-padding route.
    const auto padded_kv = is_padding ? std::min(p.sl_kv, p.n_padding) : p.sl_kv;

    // ARK drift: Neural Speed adjusts these under NS_TP_MODEL; ARK has no TP.
    const int32_t k_offset = 0;
    const int32_t log_head_num = p.head_num;

    // alibi slope
    const int n_heads_log2_floor = 1 << static_cast<int>(floor(log2(log_head_num)));
    const float m0 = powf(2.0f, -(8.f) / n_heads_log2_floor);         // 8.f is a param of alibi but hardcode now
    const float m1 = powf(2.0f, -(8.f / 2.0f) / n_heads_log2_floor);  // 8.f is a param of alibi but hardcode now
    const float tanh_scale = is_tanh ? 30.f : 0.f;                    // 30.f is a param of tanh but hardcode now

    const auto m_tiles = utils::updiv(p.sl_q, M_TILE);
    const auto num_tasks = num_heads * m_tiles;

    using Scheduler2D = bestla::parallel::Scheduler2D;
    const Scheduler2D parl({th.num_threads(), {num_tasks, 1}, {1, 1}, {0, 0}});  // main parallel scheduler

    th.parallel_for([&](int tid) {
      const int tmp_s_size = M_TILE * utils::padto(utils::padto(p.sl_kv, GemmQK::NTILE), GemmPV::KTILE);
      const int tmp_bytes = tmp_s_size * sizeof(float);  // S & exp
      const auto tmp_s = reinterpret_cast<float*>(p.tmp + tid * tmp_bytes);
      using PType = typename GemmPV::AType;
      const auto tmp_p = reinterpret_cast<PType*>(tmp_s);  // overwrite tmp_s row-wisely

      // calculate mm + softmax + mm
      {
        typename parallel::ThreadProblem2D thdp{tid};
        parl.getIndex(thdp);
        const auto [task_start, _assert0] = thdp.loc;
        auto [task_size, _assert_max1] = thdp.size;
        assert(task_size == 0 || _assert0 == 0);
        assert(task_size == 0 || _assert_max1 == 1 || _assert_max1 == 0);
        if (_assert_max1 == 0 || !thdp.valid) task_size = 0;

        for (int task_id = task_start; task_id < task_start + task_size; ++task_id) {
          const int ibat = task_id / m_tiles;
          const int i_m = task_id % m_tiles * M_TILE;
          const int ibs = ibat / p.head_num;
          const int ihn = ibat % p.head_num;
          const int ihkv = ihn / group_heads;  // GQA mapping
          const int m_size = std::min(M_TILE, p.sl_q - i_m);

          const auto alibi_ihn_m = !is_alibi ? 0.f
                                   : (ihn + k_offset < n_heads_log2_floor)
                                       ? powf(m0, ihn + k_offset + 1)
                                       : powf(m1, 2 * (ihn + k_offset - n_heads_log2_floor) + 1);

          float s_max[M_TILE]{};  // maximum for each row of the S matrix
          std::fill_n(s_max, M_TILE, -INFINITY);

          // ptr to Q / K / V / dst matrix of the current head (stride-driven)
          const auto head_q = p.Q + ibs * p.step_q_bs + ihn * p.step_q_head_num;
          const auto head_k = p.K + ibs * p.step_k_bs + ihkv * p.step_k_head_num;
          const auto head_v = p.V + ibs * p.step_v_bs + ihkv * p.step_v_head_num;
          const auto head_dst = p.dst + ibs * p.step_dst_bs + ihn * p.step_dst_head_num;
          const auto unmasked_size = is_causal ? std::min(p.sl_kv, sl_diff + i_m + M_TILE - 1 + 1)
                                     : is_padding ? padded_kv
                                                  : p.sl_kv;

          const auto unmasked_size_pad_qk = std::min(p.sl_kv, utils::padto(unmasked_size, GemmQK::NTILE));
          const auto unmasked_size_pad_pv = std::min(p.sl_kv, utils::padto(unmasked_size, GemmPV::KTILE));
          const int ld_tmp_s = utils::padto(utils::padto(unmasked_size_pad_pv, GemmQK::NTILE), GemmPV::KTILE);
          static_assert(sizeof(float) >= sizeof(PType), "PType exceeded float size!");
          const int ld_tmp_p = ld_tmp_s * sizeof(float) / sizeof(PType);
          const auto qk_prok_ldb = p.step_k_sl == 1                                 ? p.step_k_head_size
                                   : p.K_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4 ? p.step_k_sl
                                   : p.K_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2 ? p.step_k_sl
                                   : p.K_layout == ATTN_FWD_LAYOUT_NTILE24_ROWPACK1 ? p.step_k_sl
                                                                                    : (assert(0), 0);

          typename parallel::gemm::ThreadProblemBase tpQK{
              /* ThreadProblem2D */ {tid, {}, {i_m, 0}, {m_size, unmasked_size_pad_qk}, true},
              /* .block = */ {M_TILE, GemmQK::NTILE, p.head_size},
              /* .stacksize = */ _cd->getL2CacheSize(),
              /* .tmpcachesize = */ _cd->getL2CacheSize(),
          };
          l_qk.run(  // QxK => S ==exp==> P
              QKArgs{
                  utils::GemmProblem{
                      /* .batch */ 1,
                      /* .M = */ p.sl_q,
                      /* .N = */ unmasked_size_pad_qk,
                      /* .K = */ p.head_size,
                  },
                  /* .paramA = */
                  QKProQArgs{
                      head_q,
                      p.step_q_sl,
                  },
                  /* .paramB = */
                  QKProKArgs{
                      /* .B = */ head_k,
                      /* .ldb = */ qk_prok_ldb,
                      /* .is_padded = */ true,
                  },  // K should be pre-transposed
                  /* .paramC = */
                  QKEpiArgs{
                      /* .dst = */ tmp_s - i_m * ld_tmp_s,  // pretend that there is a whole S mat
                      /* .dst_max = */ s_max - i_m,         // pretend that there is a whole S mat
                      /* .ld_dst = */ ld_tmp_s,
                      /* .scale = */ p.QK_scale * p.Q_sc * p.K_sc / (tanh_scale == 0 ? 1.0f : tanh_scale),
                      // ARK: padding_type encodes the mask mode; causal reuses
                      // sl_diff, right-padding reuses the n_padding boundary.
                      /* .causal_offset = */ is_causal ? sl_diff : (is_padding ? padded_kv : -1),
                      /* .alibi_slope = */ alibi_ihn_m,
                      /* .tanh_scale = */ tanh_scale,
                      /* .padding_type = */ is_causal ? 1 : (is_padding ? 2 : 0),
                  },
              },
              tpQK);

          // softmax (with pre-computed row_max)
          const auto unmasked_size_start = is_causal ? std::min(sl_diff + i_m + 1, p.sl_kv)
                                           : is_padding ? padded_kv
                                                        : p.sl_kv;
          float expsum[M_TILE]{};  // sum of exp for each row of the S matrix
          const auto softmax_npad_size = utils::padto(unmasked_size_pad_pv, GemmPV::KTILE);
          inplace_precompute_max_softmax_t<float, PType>::template forward<RT_ISA>(  //
              m_size, unmasked_size_start, softmax_npad_size,                        // m / n
              is_causal, tmp_s, tmp_p, s_max, expsum, ld_tmp_s, ld_tmp_p);           //

          const auto pv_scale = expsum;
          // PV scale composition: V_sc / dst_sc (with the int8 1/UINT8_MAX
          // dequant factor scaffolded in, matching Neural Speed).
          for (int i = 0; i < M_TILE; ++i) pv_scale[i] = p.V_sc / UINT8_MAX / expsum[i] / p.dst_sc;

          const auto pv_prov_ldb = p.step_v_head_size == 1                          ? p.step_v_sl
                                   : p.V_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4 ? p.step_v_head_size
                                   : p.V_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2 ? p.step_v_head_size
                                   : p.V_layout == ATTN_FWD_LAYOUT_NTILE24_ROWPACK1 ? p.step_v_head_size
                                                                                    : (assert(0), 0);

          typename parallel::gemm::ThreadProblemBase tpPV{
              /* ThreadProblem2D */ {tid, {}, {0, 0}, {m_size, p.head_size}, true},
              /* .block = */ {M_TILE, GemmPV::NTILE, unmasked_size_pad_pv},
              /* .stacksize = */ _cd->getL2CacheSize(),
              /* .tmpcachesize = */ _cd->getL2CacheSize(),
          };
          l_pv.run(  // PxV => O
              PVArgs{
                  utils::GemmProblem{
                      /* .batch */ 1,
                      /* .M = */ std::min(p.sl_q - i_m, M_TILE),
                      /* .N = */ p.head_size,
                      /* .K = */ unmasked_size_pad_pv,
                  },
                  /* .paramA = */ PVProPArgs{tmp_p, ld_tmp_p},
                  /* .paramB = */
                  PVProVArgs{
                      /* .B = */ head_v,
                      /* .ldb = */ pv_prov_ldb,
                      /* .is_padded = */ true,
                  },
                  /* .paramC = */
                  composeEpiArgs<PVEpiArgs, std::is_same<V_T, int8_t>::value>(  //
                      pv_scale, head_dst + i_m * p.step_dst_sl, p.step_dst_sl),
              },
              tpPV);
        }
      }
    });
    return BTLA_CODE::Success;
  }

 protected:
  L_Max l_qk;
  L_Scale l_pv;
};

// ---------------------------------------------------------------------------
// Dtype-specialized attention dispatch (port of Neural Speed's
// `bestla_fusion_attn_forward`). The generic template is deleted so an
// unsupported Q/K/V/dst combination is a compile-time error; each supported
// combination is provided as an explicit specialization below.
//
// ARK drift vs Neural Speed (see file header):
//   * Neural Speed reaches a process-global thread pool through
//     `ne_threading::get()` and takes no threading argument. ARK has no such
//     global, so the overloads take an explicit `parallel::IThreading&` (the
//     same object `mha_stable_interface_t::compute` already consumes) and
//     forward it to `compute`.
//   * Only the fp32-score paths that compose `mha_stable_interface_t` are wired
//     here. Neural Speed's bf16/bf16, fp16/fp16 and int8 overloads (and the
//     AVX512-FP16 / AMX-BF16 ExpSum sub-paths) rely on the non-stable
//     `mha_interface_t` / `ScaleExpAccSumFp32Bf16` / avx512fp16 core, none of
//     which is migrated yet; those routes throw std::runtime_error as
//     scaffolding so an unsupported dtype/layout/ISA dispatch fails loudly
//     instead of silently no-opping in release builds (NDEBUG drops assert()).
//     The wired fp16/bf16 specializations additionally expect packed/reordered
//     (NTILE24/NTILE48) K/V and throw for raw PLAIN K/V until Phase 4.
// ---------------------------------------------------------------------------
template <typename Q_T, typename K_T, typename V_T, typename DST_T>
inline void bestla_fusion_attn_forward(const attn_fwd_args_t<Q_T, K_T, V_T, DST_T>& params,
                                       parallel::IThreading& th) = delete;

// fp32 Q, fp16 K/V (NTILE24 row-packed), fp32 dst. ARK wires only the AVX2
// stable-interface branch: Neural Speed's AVX512-FP16 branch needs the
// avx512fp16 GemmCore and its AMX-BF16 branch needs the non-stable
// `mha_interface_t` / `ScaleExpAccSumFp32Bf16`, neither migrated yet.
template <>
inline void bestla_fusion_attn_forward<float, utils::fp16, utils::fp16, float>(
    const attn_fwd_args_t<float, utils::fp16, utils::fp16, float>& params, parallel::IThreading& th) {
  GetCPUDevice();
  if (_cd->AVX2() &&                                          //
      params.K_layout == ATTN_FWD_LAYOUT_NTILE24_ROWPACK1 &&  //
      params.V_layout == ATTN_FWD_LAYOUT_NTILE24_ROWPACK1) {
#if CompileAVX2()
    using GemmKernelTrackMax = launcher_base_weight_t<  //
        gemm::SCoreRowNAvx2<24, 4>,                     //
        prologue_a::gemm::ActivationBase,               //
        weight_cvt_f16_n_tile24_t,                      //
        ScaleTrackMaxFp32Fp32>;                         //
    using GemmKernelId = launcher_base_weight_t<        //
        gemm::SCoreRowNAvx2<24, 4>,                     //
        activation_identity_t,                          // enough padding for the P-matrix
        weight_cvt_f16_n_tile24_t,                      //
        epilogue::gemm::AccumulatorWriteBackFp32>;      //
    static mha_stable_interface_t<GemmKernelTrackMax, GemmKernelId> mha;
    [[maybe_unused]] const auto ret = mha.compute(params, th);
    assert(ret == BTLA_CODE::Success);
#else
    throw std::runtime_error(
        "ark::cpu::bestla_fusion_attn_forward: fp32/fp16 attention requires an AVX2 build "
        "(CompileAVX2 disabled)");
#endif
  } else {
    throw std::runtime_error(
        "ark::cpu::bestla_fusion_attn_forward: fp32 Q + fp16 K/V is only wired for AVX2 CPUs with "
        "NTILE24 row-packed K/V; raw PLAIN (HND/NHD) K/V is not supported yet (Phase 4)");
  }
}

// fp32 Q, bf16 K/V, fp32 dst. Both the AVX512F (bf16->fp32 N-tile-48 convert)
// and AMX-BF16 (already-laid-out N-tile-48 forward) stable-interface branches
// are wired; selection mirrors Neural Speed's PREFER_FP32 gating.
template <>
inline void bestla_fusion_attn_forward<float, utils::bf16, utils::bf16, float>(
    const attn_fwd_args_t<float, utils::bf16, utils::bf16, float>& params, parallel::IThreading& th) {
  GetCPUDevice();
  if (_cd->AVX512F() &&
      ((_cd->AMX_BF16() && (params.attn_flags & ATTN_FLAG_PREFER_FP32) != 0) || !_cd->AMX_BF16())) {
#if CompileAVX512F()
    using GemmKernelBF16TrackMax = launcher_base_weight_t<  //
        gemm::SCoreRowNAvx512f<48, 8>,                      //
        prologue_a::gemm::ActivationBase,                   //
        weight_cvt_bf16_ntile48_t,                          //
        ScaleTrackMaxFp32Fp32>;                             //
    using GemmKernelBF16 = launcher_base_weight_t<          //
        gemm::SCoreRowNAvx512f<48, 8>,                      //
        activation_identity_t,                              // enough padding for the P-matrix
        weight_cvt_bf16_ntile48_t,                          //
        epilogue::gemm::AccumulatorWriteBackFp32>;          //
    static mha_stable_interface_t<GemmKernelBF16TrackMax, GemmKernelBF16> mha;
    [[maybe_unused]] const auto ret = mha.compute(params, th);
    assert(ret == BTLA_CODE::Success);
#else
    throw std::runtime_error(
        "ark::cpu::bestla_fusion_attn_forward: fp32/bf16 attention requires an AVX512F build "
        "(CompileAVX512F disabled)");
#endif
  } else if (_cd->AMX_BF16()) {
#if CompileBF16()
    using GemmKernelBF16TrackMax = launcher_base_weight_t<  //
        gemm::HCoreRowNAmxbf16<48, 16>,                     //
        prologue_a::gemm::ActivationConverterFp32,          //
        weight_forward_n_tile48_t,                          //
        ScaleTrackMaxFp32Fp32>;                             //
    using GemmKernelBF16 = launcher_base_weight_t<          //
        gemm::HCoreRowNAmxbf16<48, 16>,                     //
        activation_identity_t,                              // enough padding for the P-matrix
        weight_forward_n_tile48_t,                          //
        epilogue::gemm::AccumulatorWriteBackFp32>;          //
    static mha_stable_interface_t<GemmKernelBF16TrackMax, GemmKernelBF16> mha;
    [[maybe_unused]] const auto ret = mha.compute(params, th);
    assert(ret == BTLA_CODE::Success);
#else
    throw std::runtime_error(
        "ark::cpu::bestla_fusion_attn_forward: fp32/bf16 AMX attention requires an AMX-BF16 build "
        "(CompileBF16 disabled)");
#endif
  } else {
    throw std::runtime_error(
        "ark::cpu::bestla_fusion_attn_forward: fp32 Q + bf16 K/V requires an AVX512F or AMX-BF16 CPU "
        "with NTILE48 row-packed K/V; raw PLAIN (HND/NHD) K/V is not supported yet (Phase 4)");
  }
}

// ---------------------------------------------------------------------------
// Phase 4.5, step 1: homogeneous FP16/BF16 attention (Q == K == V == dst element
// type). Neural Speed routes these through the *non-stable* `mha_interface_t`
// (single-pass QK*V with an ExpSum epilogue folding the softmax denominator into
// the PV accumulation), not the two-pass `mha_stable_interface_t` migrated above.
// That launcher and its `ScaleExpAccSumFp32` epilogue composition are not
// migrated yet, so these specializations are documented throwing scaffolding:
// they make the homogeneous operand-type surface explicit (instead of the hard
// `= delete` on the generic primary template) and fail loudly, so a homogeneous
// dispatch cannot silently no-op in a release build. Runtime dispatch
// (sdpa.cpp / ark.cpp) still does NOT reach these overloads.
// ---------------------------------------------------------------------------

// fp16 Q/K/V, fp16 dst. Target: BestLA `gemm::HCoreRowNAvx512fp16` (native fp16
// A/B/C, ISA AVX512-FP16) driven by the non-stable `mha_interface_t` with a
// `kernel::wrapper::ScaleExpAccSumFp32<utils::fp16>` QK epilogue.
template <>
inline void bestla_fusion_attn_forward<utils::fp16, utils::fp16, utils::fp16, utils::fp16>(
    const attn_fwd_args_t<utils::fp16, utils::fp16, utils::fp16, utils::fp16>& params, parallel::IThreading& th) {
  (void)params;
  (void)th;
  throw std::runtime_error(
      "ark::cpu::bestla_fusion_attn_forward: homogeneous fp16 attention is not implemented yet (Phase 4.5): it "
      "needs the non-stable mha_interface_t launcher over gemm::HCoreRowNAvx512fp16 with a ScaleExpAccSumFp32<fp16> "
      "epilogue, neither migrated yet");
}

// bf16 Q/K/V, bf16 dst. Target: AMX-BF16 `gemm::HCoreRowNAmxbf16` core driven by
// the non-stable `mha_interface_t` with a `ScaleExpAccSumFp32<utils::bf16>`
// (avx512_bf16 sub-path) QK epilogue.
template <>
inline void bestla_fusion_attn_forward<utils::bf16, utils::bf16, utils::bf16, utils::bf16>(
    const attn_fwd_args_t<utils::bf16, utils::bf16, utils::bf16, utils::bf16>& params, parallel::IThreading& th) {
  (void)params;
  (void)th;
  throw std::runtime_error(
      "ark::cpu::bestla_fusion_attn_forward: homogeneous bf16 attention is not implemented yet (Phase 4.5): it "
      "needs the non-stable mha_interface_t launcher over gemm::HCoreRowNAmxbf16 with a ScaleExpAccSumFp32<bf16> "
      "epilogue, neither migrated yet");
}

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

// Phase 4.5 step 1: homogeneous low-precision GemmCores. These pin the cores the
// homogeneous fp16/bf16 `bestla_fusion_attn_forward` overloads will drive so
// they are type-checked / compiled at this step; the non-stable mha_interface_t
// launcher and its ScaleExpAccSumFp32 epilogue composition are deferred.
// avx512fp16 core (native fp16 A/B/C) for homogeneous fp16 attention.
using CoreAvx512Fp16 = gemm::HCoreRowNAvx512fp16<64, 0>;
// AMX bf16 core reused for homogeneous bf16 attention (same core, ExpSum path).
using CoreAmxBf16Homogeneous = gemm::HCoreRowNAmxbf16<48, 16>;
// ExpSum QK epilogues the non-stable interface will compose for each dtype.
using ScaleExpAccSumFp16 = kernel::wrapper::ScaleExpAccSumFp32<utils::fp16>;
using ScaleExpAccSumBf16 = kernel::wrapper::ScaleExpAccSumFp32<utils::bf16>;

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

// ---------------------------------------------------------------------------
// Stable-interface syntax-checks. Each pins mha_stable_interface_t to a concrete
// QK (track-max) + PV (write-back) launcher pair so compute() is fully
// type-checked / compiled here. These compositions are exactly the launcher
// pairs the `bestla_fusion_attn_forward` overloads above instantiate (AVX2 for
// `<float, fp16, fp16, float>`; AVX512F / AMX-BF16 for `<float, bf16, bf16,
// float>`); they are retained as ISA-agnostic compile pins independent of the
// runtime CPU-feature dispatch.
// ---------------------------------------------------------------------------

// AVX2: SCoreRowNAvx2<24, 4> path (fp32 scores, fp16->fp32 N-tile-24 weights).
using QKTrackMaxAvx2 = launcher_base_weight_t<CoreAvx2, prologue_a::gemm::ActivationBase, weight_cvt_f16_n_tile24_t,
                                              ScaleTrackMaxFp32Fp32>;
using PVWriteBackAvx2 = launcher_base_weight_t<CoreAvx2, activation_identity_t, weight_cvt_f16_n_tile24_t,
                                               epilogue::gemm::AccumulatorWriteBackFp32>;
using MhaStableAvx2 = mha_stable_interface_t<QKTrackMaxAvx2, PVWriteBackAvx2>;

// AVX512F: SCoreRowNAvx512f<48, 8> path (fp32 scores, bf16->fp32 N-tile-48).
using QKTrackMaxAvx512f = launcher_base_weight_t<CoreAvx512f, prologue_a::gemm::ActivationBase,
                                                 weight_cvt_bf16_ntile48_t, ScaleTrackMaxFp32Fp32>;
using PVWriteBackAvx512f = launcher_base_weight_t<CoreAvx512f, activation_identity_t, weight_cvt_bf16_ntile48_t,
                                                  epilogue::gemm::AccumulatorWriteBackFp32>;
using MhaStableAvx512f = mha_stable_interface_t<QKTrackMaxAvx512f, PVWriteBackAvx512f>;

// AMX BF16: HCoreRowNAmxbf16<48, 16> path (already-laid-out N-tile-48 weights).
using QKTrackMaxAmxBf16 = launcher_base_weight_t<CoreAmxBf16, prologue_a::gemm::ActivationConverterFp32,
                                                 weight_forward_n_tile48_t, ScaleTrackMaxFp32Fp32>;
using PVWriteBackAmxBf16 = launcher_base_weight_t<CoreAmxBf16, activation_identity_t, weight_forward_n_tile48_t,
                                                  epilogue::gemm::AccumulatorWriteBackFp32>;
using MhaStableAmxBf16 = mha_stable_interface_t<QKTrackMaxAmxBf16, PVWriteBackAmxBf16>;
}  // namespace instantiation_check

}  // namespace bestla_mha
}  // namespace ark::cpu
